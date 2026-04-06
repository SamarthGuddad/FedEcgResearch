"""
Pretrain the global 1D CNN model on a small seed dataset.

Trains on a balanced 10% seed drawn from all hospitals to give the
global model a reasonable starting point before federation begins.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server.global_model import get_model, CLASS_NAMES

# ── Configuration ──────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
PRETRAINED_PATH = os.path.join(CHECKPOINT_DIR, "global_model_pretrained.pt")
SEED_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pretrain():
    """
    Pretrain the global model on the seed dataset for SEED_EPOCHS epochs.
    Saves the trained model to global_model_pretrained.pt.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Check if already pretrained
    if os.path.exists(PRETRAINED_PATH):
        print(f"[INFO] Pretrained model already exists at {PRETRAINED_PATH}")
        return PRETRAINED_PATH

    # Load seed data
    seed_X_path = os.path.join(PROCESSED_DIR, "seed_X.npy")
    seed_y_path = os.path.join(PROCESSED_DIR, "seed_y.npy")

    if not os.path.exists(seed_X_path):
        print("[ERROR] Seed dataset not found. Run data/download_data.py first.")
        sys.exit(1)

    seed_X = np.load(seed_X_path)
    seed_y = np.load(seed_y_path)
    print(f"[INFO] Loaded seed dataset: {seed_X.shape[0]} samples")

    # Create PyTorch dataset
    # Reshape X to (N, 1, 1800) for Conv1d input
    X_tensor = torch.from_numpy(seed_X).float().unsqueeze(1)  # (N, 1, 1800)
    y_tensor = torch.from_numpy(seed_y).long()

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # Initialize model
    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[INFO] Starting pretraining on {DEVICE} for {SEED_EPOCHS} epochs...")
    print(f"       Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")

    for epoch in range(1, SEED_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        print(f"  Epoch [{epoch}/{SEED_EPOCHS}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss

    # Save pretrained model
    torch.save(model.state_dict(), PRETRAINED_PATH)
    print(f"\n[INFO] Pretrained model saved to {PRETRAINED_PATH}")
    print(f"       Final Loss: {epoch_loss:.4f} | Final Acc: {epoch_acc:.1f}%")

    return PRETRAINED_PATH


if __name__ == "__main__":
    pretrain()
