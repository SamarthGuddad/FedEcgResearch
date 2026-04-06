"""
Local training loop for hospital clients.

Handles loading the global model weights, training on local data,
and returning updated weights — never exposing raw data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


DEVICE = "cpu"  # Clients run on CPU for simulation


class LocalTrainer:
    """
    Manages the local training loop for a hospital client.

    Trains a local copy of the model on the hospital's private data,
    then returns the updated model weights (never the data).
    """

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        """
        Args:
            model: The neural network model (initialized with global weights).
            learning_rate: Learning rate for local SGD.
        """
        self.model = model.to(DEVICE)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, num_epochs: int = 5) -> Tuple[float, float]:
        """
        Perform local training for the specified number of epochs.

        Args:
            train_loader: PyTorch DataLoader with the hospital's private data.
            num_epochs: Number of local training epochs.

        Returns:
            Tuple of (average_loss, accuracy, total_samples_trained) from the final epoch.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        final_loss = 0.0
        final_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (batch_X, batch_y) in enumerate(train_loader):
                # Speed-up hack: Limit batches to reduce CPU training time to ~1 minute
                if i >= 30:
                    break

                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            epoch_loss = running_loss / total if total > 0 else 0.0
            epoch_acc = 100.0 * correct / total if total > 0 else 0.0
            final_loss = epoch_loss
            final_acc = epoch_acc

        return final_loss, final_acc, total

    def evaluate(self, test_loader) -> Tuple[float, float]:
        """
        Evaluate the model on a test set.

        Args:
            test_loader: PyTorch DataLoader for evaluation.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                running_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def get_state_dict(self) -> dict:
        """Return the model's current state_dict (weights only, no data)."""
        return self.model.state_dict()
