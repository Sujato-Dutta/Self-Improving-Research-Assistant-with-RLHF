import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.config import config
from src.reward.model import LightweightRewardModel
from src.reward.dataset import PreferencePairDataset

logger = logging.getLogger(__name__)


def bradley_terry_loss(
    r_chosen: torch.Tensor,
    r_rejected: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pairwise Bradley-Terry preference loss: -log(sigmoid(r_chosen - r_rejected))."""
    logits = r_chosen - r_rejected
    loss = -torch.nn.functional.logsigmoid(logits)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


class RewardModelTrainer:
    """Trains LightweightRewardModel using Bradley-Terry pairwise preference loss."""

    def __init__(
        self,
        model: Optional[LightweightRewardModel] = None,
        learning_rate: float = config.reward.learning_rate,
        device: str = config.model.device
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = (model or LightweightRewardModel()).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config.reward.weight_decay
        )

    def train(
        self,
        dataset: PreferencePairDataset,
        epochs: int = config.reward.epochs,
        batch_size: int = config.reward.batch_size,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        if len(dataset) == 0:
            logger.warning("Empty dataset provided to RewardModelTrainer.")
            return {"loss": 0.0, "epochs": 0}

        val_size = max(1, int(len(dataset) * 0.15)) if len(dataset) > 6 else 0
        train_size = len(dataset) - val_size

        if val_size > 0:
            train_set, val_set = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        else:
            train_set = dataset
            val_loader = None

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                q_emb = batch["query_emb"].to(self.device)
                c_emb = batch["chosen_emb"].to(self.device)
                r_emb = batch["rejected_emb"].to(self.device)
                c_feats = batch["chosen_feats"].to(self.device)
                r_feats = batch["rejected_feats"].to(self.device)
                weights = batch["weight"].to(self.device)

                self.optimizer.zero_grad()
                r_chosen = self.model(q_emb, c_emb, c_feats)
                r_rejected = self.model(q_emb, r_emb, r_feats)

                loss = bradley_terry_loss(r_chosen, r_rejected, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_train_loss += loss.item() * len(q_emb)

            avg_train_loss = total_train_loss / max(1, train_size)
            history["train_loss"].append(avg_train_loss)

            # Validation step
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_loader(val_loader)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f}")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), str(save_path))
            logger.info(f"Saved reward model weights to: {save_path}")

        return history

    def _evaluate_loader(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                q_emb = batch["query_emb"].to(self.device)
                c_emb = batch["chosen_emb"].to(self.device)
                r_emb = batch["rejected_emb"].to(self.device)
                c_feats = batch["chosen_feats"].to(self.device)
                r_feats = batch["rejected_feats"].to(self.device)
                weights = batch["weight"].to(self.device)

                r_chosen = self.model(q_emb, c_emb, c_feats)
                r_rejected = self.model(q_emb, r_emb, r_feats)

                loss = bradley_terry_loss(r_chosen, r_rejected, weights)
                total_loss += loss.item() * len(q_emb)
                correct += (r_chosen > r_rejected).sum().item()
                total += len(q_emb)

        return (total_loss / max(1, total)), (correct / max(1, total))
