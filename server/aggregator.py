"""
Federated Averaging (FedAvg) Aggregator.

Implements weighted averaging of model parameters from multiple hospital
clients, proportional to their local dataset sizes.
"""

import copy
import torch
import numpy as np
from typing import Dict, List, Tuple


class FedAvgAggregator:
    """
    FedAvg aggregation engine.

    Collects model weight submissions from clients and aggregates them
    using a weighted average based on each client's number of training samples.
    """

    def __init__(self):
        self.received_weights: Dict[str, Tuple[dict, int]] = {}
        # Maps client_id → (state_dict, num_samples)

    def submit(self, client_id: str, state_dict: dict, num_samples: int):
        """
        Register a client's updated weights after local training.

        Args:
            client_id: Unique identifier for the hospital client.
            state_dict: Model state_dict (tensors) after local training.
            num_samples: Number of samples the client trained on.
        """
        self.received_weights[client_id] = (copy.deepcopy(state_dict), num_samples)

    def aggregate(self, global_state_dict: dict) -> dict:
        """
        Perform FedAvg aggregation across all submitted client weights.

        new_global_weight[layer] = sum(client_i_weight[layer] * n_i) / sum(n_i)

        Args:
            global_state_dict: Current global model state_dict (used as template).

        Returns:
            Aggregated state_dict with updated weights.
        """
        if len(self.received_weights) == 0:
            print("[WARN] No client weights received. Returning current global model.")
            return copy.deepcopy(global_state_dict)

        # Calculate total samples across all clients
        total_samples = sum(n for _, n in self.received_weights.values())

        if total_samples == 0:
            print("[WARN] Total samples is 0. Returning current global model.")
            return copy.deepcopy(global_state_dict)

        # Initialize aggregated state dict with zeros
        aggregated = {}
        layer_names = list(global_state_dict.keys())

        for layer_name in layer_names:
            aggregated[layer_name] = torch.zeros_like(
                global_state_dict[layer_name], dtype=torch.float32
            )

        # Weighted sum
        for client_id, (client_sd, n_samples) in self.received_weights.items():
            weight = n_samples / total_samples
            for layer_name in layer_names:
                if layer_name in client_sd:
                    aggregated[layer_name] += client_sd[layer_name].float() * weight
                else:
                    print(f"[WARN] Layer {layer_name} missing from client {client_id}")

        # Clear received weights buffer for next round
        self.received_weights.clear()

        return aggregated

    def get_num_submissions(self) -> int:
        """Return the number of client submissions received."""
        return len(self.received_weights)

    def get_submitted_clients(self) -> List[str]:
        """Return list of client IDs that have submitted."""
        return list(self.received_weights.keys())

    def compute_weight_divergence(self, global_state_dict: dict) -> Dict[str, float]:
        """
        Compute L2 divergence between each client's weights and the global model.

        Useful for monitoring how much each client's local training diverges
        from the global consensus (potential indicator of data heterogeneity).

        Returns:
            Dict mapping client_id → L2 divergence score.
        """
        divergences = {}
        for client_id, (client_sd, _) in self.received_weights.items():
            total_div = 0.0
            n_params = 0
            for layer_name in global_state_dict:
                if layer_name in client_sd:
                    diff = (client_sd[layer_name].float() - global_state_dict[layer_name].float())
                    total_div += torch.norm(diff).item() ** 2
                    n_params += diff.numel()
            divergences[client_id] = np.sqrt(total_div / max(n_params, 1))
        return divergences
