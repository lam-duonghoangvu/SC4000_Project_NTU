"""Binary "same scene" classifier on top of frozen global descriptors."""

from pathlib import Path

import torch
import torch.nn as nn


class PairClassifierHead(nn.Module):
    """logit = MLP([pa, pb, pa * pb, |pa - pb|]) with pa, pb a shared projection of the two descriptors.

    The output is symmetrised, (f(a, b) + f(b, a)) / 2, so the score does not depend on pair order.
    """

    def __init__(self, input_dimension: int, projection_dimension: int = 256, hidden_dimension: int = 256, dropout: float = 0.1):
        super().__init__()
        self.head_config = {
            "input_dimension": input_dimension,
            "projection_dimension": projection_dimension,
            "hidden_dimension": hidden_dimension,
            "dropout": dropout,
        }
        self.projection = nn.Sequential(nn.Linear(input_dimension, projection_dimension), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Linear(4 * projection_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def _ordered_logit(self, projected_a: torch.Tensor, projected_b: torch.Tensor) -> torch.Tensor:
        pair_features = torch.cat(
            [projected_a, projected_b, projected_a * projected_b, (projected_a - projected_b).abs()], dim=-1
        )
        return self.classifier(pair_features).squeeze(-1)

    def forward(self, descriptor_a: torch.Tensor, descriptor_b: torch.Tensor) -> torch.Tensor:
        """Returns logits; apply sigmoid for the same-scene probability."""
        projected_a = self.projection(descriptor_a)
        projected_b = self.projection(descriptor_b)
        return 0.5 * (self._ordered_logit(projected_a, projected_b) + self._ordered_logit(projected_b, projected_a))


def save_pair_classifier_head(head: PairClassifierHead, checkpoint_path: Path, extra_metadata: dict) -> None:
    torch.save({"state_dict": head.state_dict(), "head_config": head.head_config, **extra_metadata}, checkpoint_path)


def load_pair_classifier_head(checkpoint_path: Path, device: str) -> PairClassifierHead:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    head = PairClassifierHead(**checkpoint["head_config"])
    head.load_state_dict(checkpoint["state_dict"])
    return head.eval().to(device)
