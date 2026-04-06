"""
Simulated Hospital Client for Federated Learning.

Each client:
1. Pulls global model weights from the server
2. Loads them into a local model copy
3. Trains locally on its private data partition
4. Sends updated weights (never raw data) back to the server
"""

import os
import sys
import json
import requests
import torch

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server.global_model import get_model, serialize_state_dict, deserialize_state_dict
from client.data_loader import HospitalDataLoader
from client.local_trainer import LocalTrainer


class FederatedClient:
    """
    A federated learning client representing a single hospital.

    Maintains strict data isolation: the client's raw ECG data never
    leaves this class. Only model weights are communicated to the server.
    """

    def __init__(
        self,
        client_id: str,
        hospital_id: str,
        server_url: str = "http://127.0.0.1:5000",
        local_epochs: int = 5,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            client_id: Unique identifier for this client.
            hospital_id: Hospital partition to load (e.g., "hospital_1").
            server_url: URL of the central federation server.
            local_epochs: Number of local training epochs per round.
            batch_size: Batch size for local training.
            learning_rate: Learning rate for local optimizer.
        """
        self.client_id = client_id
        self.hospital_id = hospital_id
        self.server_url = server_url
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        # Load this hospital's private data
        self.data_loader = HospitalDataLoader(hospital_id, batch_size=batch_size)
        self.num_samples = self.data_loader.num_train_samples

        # Local model (will be loaded from global weights each round)
        self.model = get_model()

        print(f"[{self.client_id}] Initialized | Hospital: {hospital_id} | "
              f"Train samples: {self.num_samples}")

    def pull_global_model(self) -> bool:
        """
        Pull the current global model weights from the server.

        Returns:
            True if successful, False otherwise.
        """
        try:
            response = requests.get(f"{self.server_url}/get_model", timeout=60)
            if response.status_code == 200:
                data = response.json()
                weights = deserialize_state_dict(data["weights"])
                self.model.load_state_dict(weights)
                return True
            else:
                print(f"[{self.client_id}] Failed to pull model: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"[{self.client_id}] Error pulling model: {e}")
            return False

    def train_locally(self):
        """
        Train the model on local private data.

        Returns:
            Tuple of (loss, accuracy) from training.
        """
        trainer = LocalTrainer(self.model, learning_rate=self.learning_rate)
        train_loader = self.data_loader.get_train_loader()

        loss, accuracy, actual_samples = trainer.train(train_loader, num_epochs=self.local_epochs)
        self.model = trainer.model  # Update local model with trained weights

        return loss, accuracy, actual_samples

    def push_weights(self, local_loss: float, local_accuracy: float, actual_samples: int) -> bool:
        """
        Push local model weights to the server.
        Only weights are transmitted — never raw ECG data.

        Args:
            local_loss: Training loss from this round.
            local_accuracy: Training accuracy from this round.
            actual_samples: Number of samples actually trained on.

        Returns:
            True if successful, False otherwise.
        """
        try:
            state_dict = self.model.state_dict()
            serialized = serialize_state_dict(state_dict)

            payload = {
                "client_id": self.client_id,
                "weights": serialized,
                "num_samples": actual_samples,
                "local_loss": local_loss,
                "local_accuracy": local_accuracy,
            }

            response = requests.post(
                f"{self.server_url}/submit_weights",
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                return True
            else:
                print(f"[{self.client_id}] Failed to push weights: HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"[{self.client_id}] Error pushing weights: {e}")
            return False

    def run_round(self) -> dict:
        """
        Execute one complete federation round for this client:
        1. Pull global model
        2. Train locally
        3. Push updated weights

        Returns:
            Dictionary with round results.
        """
        print(f"\n[{self.client_id}] ── Starting federation round ──")

        # Step 1: Pull global model
        if not self.pull_global_model():
            return {"status": "error", "message": "Failed to pull global model"}
        print(f"[{self.client_id}] Global model pulled successfully.")

        # Step 2: Train locally
        loss, accuracy, actual_samples = self.train_locally()
        print(f"[{self.client_id}] Local training complete | "
              f"Loss: {loss:.4f} | Acc: {accuracy:.1f}%")

        # Step 3: Push updated weights
        if not self.push_weights(loss, accuracy, actual_samples):
            return {"status": "error", "message": "Failed to push weights"}
        print(f"[{self.client_id}] Weights pushed to server.")

        return {
            "status": "ok",
            "client_id": self.client_id,
            "loss": loss,
            "accuracy": accuracy,
            "num_samples": actual_samples,
        }

    def evaluate_local(self) -> dict:
        """Evaluate the current model on this hospital's test set."""
        trainer = LocalTrainer(self.model)
        test_loader = self.data_loader.get_test_loader()
        loss, accuracy = trainer.evaluate(test_loader)
        return {"loss": loss, "accuracy": accuracy}
