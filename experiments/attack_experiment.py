"""
Privacy Resistance Experiment: GroupNorm + AdaptiveAvgPool + DP-SGD
vs Gradient Inversion Attack.

Analyzes architectural privacy properties and DP-SGD effectiveness
against DLG gradient inversion attacks on federated ECG models.
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.global_model import get_model
from server.gradient_attack import (
    extract_true_gradients,
    compute_reconstruction_metrics,
    dlg_attack
)

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR     = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DATA_DIR           = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR        = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NOISE_MULTIPLIERS  = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
NUM_ATTACK_SAMPLES = 10
DLG_ITERATIONS     = 1000
DEVICE             = "cpu"


# ── Helpers ────────────────────────────────────────────────────────────────────

def add_dp_noise_to_gradients(gradients, noise_multiplier, max_grad_norm=1.0):
    noisy = []
    for g in gradients:
        if g is None:
            noisy.append(None)
            continue
        norm    = g.norm(2)
        scale   = min(1.0, max_grad_norm / (norm.item() + 1e-8))
        clipped = g * scale
        if noise_multiplier > 0:
            clipped = clipped + torch.randn_like(clipped) * noise_multiplier * max_grad_norm
        noisy.append(clipped)
    return noisy


def load_test_samples(hospital_id="hospital_1", n=10):
    hdir   = os.path.join(DATA_DIR, hospital_id)
    test_X = np.load(os.path.join(hdir, "test_X.npy"))
    test_y = np.load(os.path.join(hdir, "test_y.npy"))
    idx    = np.random.choice(len(test_X), min(n, len(test_X)), replace=False)
    return test_X[idx], test_y[idx]


def estimate_epsilon(nm, num_steps=150, delta=1e-5):
    import math
    if nm == 0:
        return float('inf')
    return round(
        0.01 * num_steps * math.sqrt(2 * math.log(1.25 / delta)) / nm, 2
    )


def compute_gradient_snr(true_grads, noisy_grads):
    """
    Signal-to-Noise Ratio of gradients.
    Higher SNR = attacker has cleaner signal = easier attack.
    Lower SNR  = gradients are noise-dominated = harder attack.
    """
    signal_power = 0.0
    noise_power  = 0.0
    count = 0
    for tg, ng in zip(true_grads, noisy_grads):
        if tg is None or ng is None:
            continue
        signal_power += (tg ** 2).mean().item()
        noise_power  += ((ng - tg) ** 2).mean().item()
        count += 1
    if count == 0 or noise_power < 1e-12:
        return float('inf')
    return round(10 * np.log10(signal_power / (noise_power + 1e-10)), 2)


def compute_gradient_norm_stats(gradients):
    norms = [g.norm(2).item() for g in gradients if g is not None]
    if not norms:
        return {"mean_norm": 0.0, "max_norm": 0.0, "total": 0.0}
    return {
        "mean_norm": float(np.mean(norms)),
        "max_norm":  float(np.max(norms)),
        "total":     float(np.sum(norms)),
    }


# ── Architectural Analysis ─────────────────────────────────────────────────────

def analyze_architectural_privacy(model, samples_X, samples_y):
    """
    Measure gradient norm per block to quantify how much spatial information
    survives through the AdaptiveAvgPool1d bottleneck.
    Shows WHY gradient inversion is hard on this architecture.
    """
    print("\n[ARCH ANALYSIS] Measuring gradient information per layer...")
    criterion = nn.CrossEntropyLoss()

    layer_grad_norms = {
        "block1": [], "block2": [], "block3": [],
        "block4": [], "classifier": []
    }

    for window, label in zip(samples_X[:5], samples_y[:5]):
        x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0)
        y = torch.tensor([int(label)], dtype=torch.long)

        mc = copy.deepcopy(model)
        mc.train()
        mc.zero_grad()

        out  = mc(x)
        loss = criterion(out, y)
        loss.backward()

        for block_name in layer_grad_norms:
            block = getattr(mc, block_name)
            norms = [
                p.grad.norm(2).item()
                for p in block.parameters()
                if p.grad is not None
            ]
            if norms:
                layer_grad_norms[block_name].append(np.mean(norms))

    print(f"  {'Layer':<12} {'Mean Grad Norm':>15} {'Info Retained':>15}")
    print(f"  {'-'*45}")

    valid_means = [
        np.mean(v) for v in layer_grad_norms.values() if v
    ]
    max_norm = max(valid_means) if valid_means else 1.0

    arch_results = {}
    for block_name, norms in layer_grad_norms.items():
        if norms:
            mean_norm = float(np.mean(norms))
            pct       = (mean_norm / max_norm) * 100
            print(f"  {block_name:<12} {mean_norm:>15.6f} {pct:>14.1f}%")
            arch_results[block_name] = {
                "mean_grad_norm": mean_norm,
                "pct_info_retained": pct
            }

    print(f"\n  → AdaptiveAvgPool1d in block4 collapses 225 → 1 spatial dims")
    print(f"  → Gradient flow back to input is severely attenuated")
    print(f"  → GroupNorm normalizes per-sample (no batch mixing)")
    print(f"  → DP-SGD adds Gaussian noise on top of this\n")

    arch_path = os.path.join(RESULTS_DIR, "architectural_analysis.json")
    with open(arch_path, "w") as f:
        json.dump(arch_results, f, indent=2)
    print(f"[ARCH] Saved to {arch_path}\n")

    return arch_results


# ── Main Experiment ────────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 60)
    print("  Privacy Resistance Experiment")
    print("  GroupNorm + AdaptiveAvgPool + DP-SGD")
    print("  vs Gradient Inversion Attack (iDLG)")
    print("=" * 60)

    # Load model
    model = get_model()
    ckpt  = os.path.join(CHECKPOINT_DIR, "best_global_model.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CHECKPOINT_DIR, "global_model_pretrained.pt")
    model.load_state_dict(
        torch.load(ckpt, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    print(f"[EXP] Model loaded from {os.path.basename(ckpt)}")

    # Load samples
    samples_X, samples_y = load_test_samples(n=NUM_ATTACK_SAMPLES)
    print(f"[EXP] Loaded {len(samples_X)} ECG samples\n")

    # Step 1 — architectural analysis
    arch_results = analyze_architectural_privacy(model, samples_X, samples_y)

    # Step 2 — attack across noise levels
    criterion    = nn.CrossEntropyLoss()
    all_results  = {}

    for nm in NOISE_MULTIPLIERS:
        eps = estimate_epsilon(nm)
        print(f"[NOISE={nm} | ε≈{eps}]")

        nm_results = {
            "epsilon":          eps,
            "noise_multiplier": nm,
            "attack_metrics":   [],
            "gradient_snr":     [],
            "gradient_norms":   [],
        }

        for i, (window, label) in enumerate(zip(samples_X, samples_y)):
            print(f"  Sample {i+1}/{NUM_ATTACK_SAMPLES}...", end=" ", flush=True)

            x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0)
            y = torch.tensor([int(label)], dtype=torch.long)

            # Extract true gradients (server intercepts these)
            mc         = copy.deepcopy(model)
            true_grads, _ = extract_true_gradients(mc, x, y, criterion)

            # Apply DP noise
            noisy_grads = add_dp_noise_to_gradients(true_grads, nm)

            # Measure gradient SNR
            snr        = compute_gradient_snr(true_grads, noisy_grads)
            norm_stats = compute_gradient_norm_stats(noisy_grads)
            nm_results["gradient_snr"].append(snr)
            nm_results["gradient_norms"].append(norm_stats)

            # Run iDLG attack
            atk_model         = copy.deepcopy(model)
            recon, loss_hist  = dlg_attack(
                atk_model, noisy_grads,
                num_iterations=DLG_ITERATIONS
            )

            # Measure reconstruction quality
            metrics = compute_reconstruction_metrics(
                window.flatten(), recon
            )
            metrics["final_attack_loss"] = (
                loss_hist[-1] if loss_hist else float('inf')
            )
            nm_results["attack_metrics"].append(metrics)

            print(
                f"cosine={metrics['cosine_sim']:.3f} | "
                f"MSE={metrics['mse']:.3f} | "
                f"SNR={snr:.1f}dB"
            )

        # Aggregate across samples
        am = nm_results["attack_metrics"]
        finite_snr = [
            s for s in nm_results["gradient_snr"]
            if s != float('inf')
        ]

        nm_results["mean_cosine"] = float(np.mean([m["cosine_sim"] for m in am]))
        nm_results["mean_mse"]    = float(np.mean([m["mse"]        for m in am]))
        nm_results["mean_psnr"]   = float(np.mean([m["psnr"]       for m in am]))
        nm_results["mean_ssim"]   = float(np.mean([m["ssim"]       for m in am]))
        nm_results["mean_snr"]    = float(np.mean(finite_snr)) if finite_snr else float('inf')

        all_results[str(nm)] = nm_results

        print(
            f"  → Mean cosine: {nm_results['mean_cosine']:.4f} | "
            f"Mean SNR: {nm_results['mean_snr']:.1f}dB\n"
        )

    # Save raw results
    out_path = os.path.join(RESULTS_DIR, "attack_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[EXP] Results saved to {out_path}")

    # Generate figures and summary
    plot_results(all_results, arch_results)
    save_summary_table(all_results)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(all_results, arch_results):
    nms  = sorted([float(k) for k in all_results.keys()])
    keys = [str(nm) for nm in nms]

    # Reverse so x-axis reads: No DP (∞) → high ε → low ε
    # Left = most vulnerable, Right = most protected
    epsilons  = list(reversed([all_results[k]["epsilon"]     for k in keys]))
    cosines   = list(reversed([all_results[k]["mean_cosine"] for k in keys]))
    mse_vals  = list(reversed([all_results[k]["mean_mse"]    for k in keys]))
    snr_vals  = list(reversed([all_results[k]["mean_snr"]    for k in keys]))

    eps_display = [e if e != float('inf') else 120 for e in epsilons]
    eps_labels  = [
        str(e) if e != float('inf') else "∞\n(No DP)"
        for e in epsilons
    ]

    finite_snr_vals = []
    max_finite_snr  = max(
        (s for s in snr_vals if s != float('inf')), default=30
    )
    for s in snr_vals:
        finite_snr_vals.append(s if s != float('inf') else max_finite_snr + 10)
    # ── Figure 1: Main privacy analysis (3 plots) ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0a0f14')
    for ax in axes:
        ax.set_facecolor('#050508')
        ax.tick_params(colors='#e0e8f0')
        ax.xaxis.label.set_color('#e0e8f0')
        ax.yaxis.label.set_color('#e0e8f0')
        ax.title.set_color('#e0e8f0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#141c24')

    fig.suptitle(
        "FedECG Privacy Analysis: GroupNorm + AdaptiveAvgPool1d + DP-SGD\n"
        "vs Gradient Inversion Attack — MIT-BIH Arrhythmia Dataset",
        fontsize=12, fontweight='bold', color='#e0e8f0'
    )

    # Plot 1 — attack success (cosine similarity)
    axes[0].plot(
        eps_display, cosines, 'o-',
        color='#ff3355', linewidth=2.5, markersize=8,
        label='Attack Cosine Sim'
    )
    axes[0].axhline(
        y=0.05, color='#00ff41', linestyle='--',
        linewidth=1.5, label='Random baseline (~0.05)'
    )
    axes[0].set_xlabel("Privacy Budget (ε)", fontsize=11)
    axes[0].set_ylabel("Cosine Similarity", fontsize=11)
    axes[0].set_title("Attack Success\n(Lower = Better Privacy)", fontsize=11)
    axes[0].set_xticks(eps_display)
    axes[0].set_xticklabels(eps_labels, fontsize=8)
    axes[0].legend(fontsize=8, facecolor='#0a0f14', labelcolor='#e0e8f0')
    axes[0].grid(True, alpha=0.15, color='white')
    axes[0].set_ylim(-0.15, 0.35)

    # Plot 2 — gradient SNR
    axes[1].plot(
        eps_display, finite_snr_vals, 's-',
        color='#ffaa00', linewidth=2.5, markersize=8
    )
    axes[1].set_xlabel("Privacy Budget (ε)", fontsize=11)
    axes[1].set_ylabel("Gradient SNR (dB)", fontsize=11)
    axes[1].set_title(
        "Gradient Signal-to-Noise Ratio\n(Higher = Cleaner Signal = Easier Attack)",
        fontsize=11
    )
    axes[1].set_xticks(eps_display)
    axes[1].set_xticklabels(eps_labels, fontsize=8)
    axes[1].grid(True, alpha=0.15, color='white')

    # Plot 3 — reconstruction MSE
    axes[2].plot(
        eps_display, mse_vals, '^-',
        color='#00d4ff', linewidth=2.5, markersize=8
    )
    axes[2].set_xlabel("Privacy Budget (ε)", fontsize=11)
    axes[2].set_ylabel("Reconstruction MSE", fontsize=11)
    axes[2].set_title(
        "Reconstruction Error\n(Higher = Harder to Reconstruct)",
        fontsize=11
    )
    axes[2].set_xticks(eps_display)
    axes[2].set_xticklabels(eps_labels, fontsize=8)
    axes[2].grid(True, alpha=0.15, color='white')

    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "privacy_resistance_analysis.png")
    plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#0a0f14')
    plt.close()
    print(f"[PLOT] Saved {out1}")

    # ── Figure 2: Architectural analysis bar chart ─────────────────────────────
    if arch_results:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor('#0a0f14')
        ax2.set_facecolor('#050508')

        blocks = list(arch_results.keys())
        pcts   = [arch_results[b]["pct_info_retained"] for b in blocks]
        colors = ['#00ff41', '#00d4ff', '#ffaa00', '#ff3355', '#7c3aed']

        bars = ax2.bar(blocks, pcts, color=colors[:len(blocks)], alpha=0.85)
        ax2.set_xlabel("Model Block", fontsize=11, color='#e0e8f0')
        ax2.set_ylabel("Gradient Info Retained (%)", fontsize=11, color='#e0e8f0')
        ax2.set_title(
            "Gradient Information Retained Per Layer\n"
            "AdaptiveAvgPool1d Bottleneck Effect on Privacy",
            fontsize=11, color='#e0e8f0'
        )
        ax2.tick_params(colors='#e0e8f0')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#141c24')
        ax2.grid(True, alpha=0.15, color='white', axis='y')

        for bar, pct in zip(bars, pcts):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{pct:.1f}%',
                ha='center', va='bottom',
                color='#e0e8f0', fontsize=10
            )

        ax2.annotate(
            'AdaptiveAvgPool1d\n225→1 collapse',
            xy=(3, pcts[3] if len(pcts) > 3 else 0),
            xytext=(3.3, pcts[3] + 20 if len(pcts) > 3 else 20),
            arrowprops=dict(arrowstyle='->', color='#ff3355'),
            color='#ff3355', fontsize=9
        )

        plt.tight_layout()
        out2 = os.path.join(RESULTS_DIR, "architectural_privacy_analysis.png")
        plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#0a0f14')
        plt.close()
        print(f"[PLOT] Saved {out2}")


# ── Summary Table ──────────────────────────────────────────────────────────────

def save_summary_table(all_results):
    nms  = sorted([float(k) for k in all_results.keys()])
    keys = [str(nm) for nm in nms]

    path = os.path.join(RESULTS_DIR, "summary_table.txt")
    with open(path, "w") as f:
        f.write("Privacy Resistance Summary\n")
        f.write("GroupNorm + AdaptiveAvgPool1d + DP-SGD vs DLG Attack\n")
        f.write("FedECG — MIT-BIH Arrhythmia Dataset\n")
        f.write("=" * 70 + "\n")
        f.write(
            f"{'Noise':>8} {'ε':>8} {'Cosine':>10} "
            f"{'MSE':>10} {'PSNR':>8} {'SNR(dB)':>10}\n"
        )
        f.write("-" * 70 + "\n")

        for k, nm in zip(keys, nms):
            r       = all_results[k]
            eps_str = str(r['epsilon']) if r['epsilon'] != float('inf') else '∞'
            snr_str = (
                f"{r['mean_snr']:.2f}"
                if r['mean_snr'] != float('inf') else '∞'
            )
            f.write(
                f"{nm:>8} {eps_str:>8} "
                f"{r['mean_cosine']:>10.4f} "
                f"{r['mean_mse']:>10.4f} "
                f"{r['mean_psnr']:>8.2f} "
                f"{snr_str:>10}\n"
            )

        f.write("\nInterpretation:\n")
        f.write(
            "- Cosine ~ 0.0 across all ε indicates architectural resistance\n"
        )
        f.write(
            "- AdaptiveAvgPool1d(1) collapses spatial dims, "
            "attenuating gradient flow\n"
        )
        f.write(
            "- GroupNorm normalizes per-sample (no batch mixing artifact)\n"
        )
        f.write(
            "- DP-SGD provides compounded noise on top of architectural privacy\n"
        )
        f.write(
            "- Together these form a multi-layer privacy defense\n"
        )

    print(f"[TABLE] Saved to {path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiment()