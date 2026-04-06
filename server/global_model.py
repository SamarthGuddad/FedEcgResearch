"""
1D CNN Model for ECG Arrhythmia Detection.
Shared model definition used by both server and clients.
"""

import torch
import torch.nn as nn


# Class mapping for MIT-BIH annotations → 5 arrhythmia categories
LABEL_MAP = {
    "N": 0,  # Normal beat
    "A": 1,  # Atrial Fibrillation / Supraventricular
    "V": 2,  # Premature Ventricular Contraction
    "L": 3,  # Left/Right Bundle Branch Block
    "P": 4,  # Pacemaker beat
}

CLASS_NAMES = ["Normal (N)", "Atrial Fib (A)", "PVC (V)", "BBB (L/R)", "Pacemaker (P)"]

# MIT-BIH annotation symbol → our 5-class label
ANNOTATION_TO_LABEL = {
    "N": "N", "·": "N", ".": "N",
    "L": "L", "R": "L",
    "A": "A", "a": "A", "J": "A", "S": "A", "e": "A", "j": "A",
    "V": "V", "E": "V",
    "/": "P", "f": "P", "F": "P",
}


class ECGArrhythmiaNet(nn.Module):
    """
    1D Convolutional Neural Network for ECG arrhythmia classification.

    Input:  (batch, 1, 1800)  — single-lead ECG window (5 seconds at 360 Hz)
    Output: (batch, 5)        — logits for 5 arrhythmia classes
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()

        # Layer 1: Conv1d(1, 32, 7) → BN → ReLU → MaxPool(2)
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Layer 2: Conv1d(32, 64, 5) → BN → ReLU → MaxPool(2)
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Layer 3: Conv1d(64, 128, 3) → BN → ReLU → MaxPool(2)
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Layer 4: Conv1d(128, 256, 3) → BN → ReLU → AdaptiveAvgPool(1)
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.squeeze(-1)  # (batch, 256, 1) → (batch, 256)
        x = self.classifier(x)
        return x


def get_model(num_classes: int = 5) -> ECGArrhythmiaNet:
    """Factory function to create a fresh model instance."""
    return ECGArrhythmiaNet(num_classes=num_classes)


def serialize_state_dict(state_dict: dict) -> str:
    """
    Convert state_dict to base64-encoded string using native torch.save
    for highly efficient network transfer.
    """
    import io
    import base64
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def deserialize_state_dict(serialized_str: str) -> dict:
    """Convert base64-encoded string back to PyTorch state_dict."""
    import io
    import base64
    buffer = io.BytesIO(base64.b64decode(serialized_str))
    return torch.load(buffer, map_location="cpu", weights_only=True)
