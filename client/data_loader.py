"""
Data loader for hospital clients.

Loads the pre-processed MIT-BIH partition for a specific hospital.
Each hospital only ever accesses its own data partition — never others.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class HospitalDataLoader:
    """
    Data loader for a single hospital's ECG data partition.

    Enforces data isolation: each instance only loads data belonging
    to the specified hospital.
    """

    def __init__(self, hospital_id: str, batch_size: int = 256):
        """
        Args:
            hospital_id: Hospital identifier (e.g., "hospital_1").
            batch_size: Training batch size.
        """
        self.hospital_id = hospital_id
        self.batch_size = batch_size
        self.hospital_dir = os.path.join(PROCESSED_DIR, hospital_id)

        # Validate data exists
        if not os.path.exists(self.hospital_dir):
            raise FileNotFoundError(
                f"Data for {hospital_id} not found at {self.hospital_dir}. "
                f"Run data/download_data.py first."
            )

        # Load data
        self.train_X = np.load(os.path.join(self.hospital_dir, "train_X.npy"))
        self.train_y = np.load(os.path.join(self.hospital_dir, "train_y.npy"))
        self.test_X = np.load(os.path.join(self.hospital_dir, "test_X.npy"))
        self.test_y = np.load(os.path.join(self.hospital_dir, "test_y.npy"))

        self.num_train_samples = len(self.train_X)
        self.num_test_samples = len(self.test_X)

    def get_train_loader(self) -> DataLoader:
        """Get PyTorch DataLoader for training data."""
        X = torch.from_numpy(self.train_X).float().unsqueeze(1)  # (N, 1, 1800)
        y = torch.from_numpy(self.train_y).long()
        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def get_test_loader(self) -> DataLoader:
        """Get PyTorch DataLoader for test data."""
        X = torch.from_numpy(self.test_X).float().unsqueeze(1)
        y = torch.from_numpy(self.test_y).long()
        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def get_class_distribution(self) -> dict:
        """Return class distribution in training set."""
        unique, counts = np.unique(self.train_y, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def __repr__(self):
        return (f"HospitalDataLoader({self.hospital_id}, "
                f"train={self.num_train_samples}, test={self.num_test_samples})")
