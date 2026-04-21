"""
Visualize reconstructed ECG signals vs originals at different epsilon levels.
Generates the qualitative figure for the paper showing what the attacker sees.
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.global_model import get_model
from server.gradient_attack import dlg_attack, extract_true_gradients
from experiments.attack_experiment import (
    add_dp_noise_to_gradients, load_test_samples, estimate_epsilon
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pick representative noise levels for the figure
NOISE_LEVELS_TO_SHOW = [0.0, 0.3, 1.0, 3.0]


def main():
    model = get_model()
    ckpt  = os.path.join(CHECKPOINT_DIR, "best_global_model.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CHECKPOINT_DIR, "global_model_pretrained.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    # Use a single fixed ECG sample for visual comparison
    samples_X, samples_y = load_test_samples(n=1)
    window = samples_X[0]
    label  = samples_y[0]

    x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0)
    y = torch.tensor([int(label)], dtype=torch.long)

    fig, axes = plt.subplots(
        len(NOISE_LEVELS_TO_SHOW) + 1, 1,
        figsize=(14, 3 * (len(NOISE_LEVELS_TO_SHOW) + 1))
    )
    fig.suptitle(
        "ECG Gradient Inversion Attack — Reconstruction at Different ε Levels\n"
        "FedECG | MIT-BIH Arrhythmia Database",
        fontsize=13, fontweight='bold'
    )

    t = np.arange(len(window.flatten())) / 360.0  # time axis in seconds

    # Row 0 — original signal
    axes[0].plot(t, window.flatten(), color='#00ff41', linewidth=1.2)
    axes[0].set_title("Original ECG Signal (Private — Never Shared)", fontsize=11)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_facecolor('#050508')
    axes[0].grid(True, alpha=0.2, color='white')

    criterion = nn.CrossEntropyLoss()

    for row, nm in enumerate(NOISE_LEVELS_TO_SHOW):
        eps = estimate_epsilon(nm)
        eps_str = f"ε ≈ {eps}" if eps != float('inf') else "ε = ∞ (No DP)"

        # Get gradients and apply noise
        model_copy = copy.deepcopy(model)
        true_grads, _ = extract_true_gradients(model_copy, x, y, criterion)
        noisy_grads   = add_dp_noise_to_gradients(true_grads, nm)

        # Attack
        attack_model  = copy.deepcopy(model)
        reconstructed, _ = dlg_attack(attack_model, noisy_grads, num_iterations=300)

        # Compute cosine similarity
        orig  = window.flatten()
        recon = reconstructed.flatten()[:len(orig)]
        cos   = float(np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10))

        color = '#ff3355' if nm < 0.5 else ('#ffaa00' if nm < 1.5 else '#00d4ff')
        axes[row + 1].plot(t, recon[:len(t)], color=color, linewidth=1.2)
        axes[row + 1].set_title(
            f"Reconstructed | noise_multiplier={nm} | {eps_str} | cosine_sim={cos:.3f}",
            fontsize=10
        )
        axes[row + 1].set_ylabel("Amplitude")
        axes[row + 1].set_facecolor('#050508')
        axes[row + 1].grid(True, alpha=0.2, color='white')

        attack_success = "⚠ ATTACK SUCCEEDED" if cos > 0.5 else "✅ ATTACK FAILED (DP Protected)"
        axes[row + 1].set_xlabel(attack_success, fontsize=10, color=color)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "ecg_reconstruction_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0f14')
    print(f"[VIZ] Saved to {out_path}")


if __name__ == "__main__":
    main()