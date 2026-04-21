"""
Improved Deep Leakage from Gradients (iDLG) Attack.
Uses label inference + TV regularization for 1D ECG signals.
"""

import torch
import torch.nn as nn
import numpy as np
import copy

DEVICE = "cpu"


def infer_label_from_gradients(gradients, num_classes=5):
    """
    iDLG improvement: infer the true label analytically from gradients
    before optimization. This dramatically improves attack success.
    The true label corresponds to the class whose gradient in the
    final linear layer is negative.
    """
    # Find the last gradient tensor (final linear layer bias or weight)
    last_grad = None
    for g in reversed(gradients):
        if g is not None and g.dim() <= 2:
            last_grad = g
            break

    if last_grad is None:
        return 0

    if last_grad.dim() == 1 and len(last_grad) == num_classes:
        # Bias gradient — true label is argmin
        return int(torch.argmin(last_grad).item())
    elif last_grad.dim() == 2:
        # Weight gradient — sum over input features
        return int(torch.argmin(last_grad.sum(dim=1)).item())

    return 0


def total_variation_loss_1d(signal):
    """Smoothness regularizer for 1D signals — penalizes rapid oscillations."""
    return torch.mean(torch.abs(signal[:, :, 1:] - signal[:, :, :-1]))


def dlg_attack(model, true_gradients, num_iterations=1000, lr=0.1):
    """
    Improved DLG attack for 1D ECG signals using:
    - iDLG label inference (no need to optimize label)
    - LBFGS optimizer (much better than Adam for this problem)
    - Total variation regularization (encourages smooth ECG-like signals)
    - Multiple random restarts (takes best result)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    num_classes = 5

    # Step 1 — infer label analytically (iDLG)
    inferred_label = infer_label_from_gradients(true_gradients, num_classes)
    true_label_tensor = torch.tensor([inferred_label], dtype=torch.long)

    best_loss  = float('inf')
    best_dummy = None
    loss_history = []

    # Multiple restarts to escape bad local minima
    num_restarts = 3
    for restart in range(num_restarts):
        # Initialize dummy data — try different initializations
        if restart == 0:
            dummy_data = torch.randn(1, 1, 1800, requires_grad=True, device=DEVICE)
        elif restart == 1:
            dummy_data = torch.zeros(1, 1, 1800, requires_grad=True, device=DEVICE)
        else:
            dummy_data = (torch.rand(1, 1, 1800) * 2 - 1).requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [dummy_data], lr=lr,
            max_iter=20,
            history_size=50,
            line_search_fn='strong_wolfe'
        )

        restart_losses = []

        for iteration in range(num_iterations // 20):  # LBFGS does 20 inner iters
            def closure():
                optimizer.zero_grad()

                # Forward pass with dummy data
                dummy_out  = model(dummy_data)
                dummy_loss = criterion(dummy_out, true_label_tensor)

                # Compute dummy gradients
                dummy_grad = torch.autograd.grad(
                    dummy_loss,
                    model.parameters(),
                    create_graph=True,
                    allow_unused=True
                )

                # Gradient matching loss
                grad_loss = torch.tensor(0.0, requires_grad=True)
                count = 0
                for dg, tg in zip(dummy_grad, true_gradients):
                    if dg is not None and tg is not None:
                        grad_loss = grad_loss + ((dg - tg.detach()) ** 2).sum()
                        count += 1

                if count > 0:
                    grad_loss = grad_loss / count

                # TV regularization — encourages smooth ECG-like signal
                tv_loss = total_variation_loss_1d(dummy_data)
                total_loss = grad_loss + 1e-4 * tv_loss

                total_loss.backward()
                return total_loss

            loss_val = optimizer.step(closure)
            if loss_val is not None:
                lv = loss_val.item()
                restart_losses.append(lv)

                if lv < best_loss:
                    best_loss  = lv
                    best_dummy = dummy_data.detach().clone()

        loss_history.extend(restart_losses)

    result = best_dummy.squeeze().numpy() if best_dummy is not None \
             else np.zeros(1800)
    return result, loss_history


def extract_true_gradients(model, data_sample, label, criterion=None):
    """
    Simulate what the server receives: gradients from one training step.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.train()
    model.zero_grad()

    output = model(data_sample)
    loss   = criterion(output, label)
    loss.backward()

    gradients = [
        p.grad.detach().clone() if p.grad is not None else None
        for p in model.parameters()
    ]

    return gradients, loss.item()


def compute_reconstruction_metrics(original, reconstructed):
    """Measure reconstruction quality."""
    orig  = np.array(original).flatten()
    recon = np.array(reconstructed).flatten()

    min_len = min(len(orig), len(recon))
    orig    = orig[:min_len]
    recon   = recon[:min_len]

    mse = float(np.mean((orig - recon) ** 2))

    max_val = max(np.max(np.abs(orig)), 1e-8)
    psnr    = float(10 * np.log10(max_val ** 2 / (mse + 1e-10)))

    dot      = np.dot(orig, recon)
    norm_val = (np.linalg.norm(orig) * np.linalg.norm(recon)) + 1e-10
    cosine   = float(dot / norm_val)

    mu_x  = np.mean(orig);  mu_y  = np.mean(recon)
    sig_x = np.var(orig);   sig_y = np.var(recon)
    sig_xy = np.mean((orig - mu_x) * (recon - mu_y))
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    ssim   = float(
        (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2) /
        ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))
    )

    return {"mse": mse, "psnr": psnr, "cosine_sim": cosine, "ssim": ssim}