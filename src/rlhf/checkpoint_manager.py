import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import config

logger = logging.getLogger(__name__)

ROUNDS_ORDER = ["Base", "RLHF Round 1", "Round 2", "Final"]


class CheckpointManager:
    """Manages policy model checkpoints across self-improvement rounds."""

    def __init__(self, checkpoints_dir: Path = config.checkpoints_dir):
        self.checkpoints_dir = checkpoints_dir
        self.registry_file = checkpoints_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint registry: {e}")

        initial = {
            "active_version": "Base",
            "active_path": config.model.policy_model_name,
            "rounds": {
                "Base": {
                    "round_number": 0,
                    "model_path": config.model.policy_model_name,
                    "status": "promoted",
                    "promoted_at": "initial",
                    "metrics": {
                        "win_rate": 0.50,
                        "avg_reward": 0.42,
                        "citation_accuracy": 0.74,
                        "groundedness": 0.68,
                        "hallucination_rate": 0.22
                    }
                }
            }
        }
        self._save_registry(initial)
        return initial

    def _save_registry(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save registry: {e}")

    def get_active_version(self) -> str:
        return self.registry.get("active_version", "Base")

    def get_active_model_path(self) -> str:
        return self.registry.get("active_path", config.model.policy_model_name)

    def register_candidate(
        self,
        round_name: str,
        round_number: int,
        model_path: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        self.registry["rounds"][round_name] = {
            "round_number": round_number,
            "model_path": model_path,
            "status": "candidate",
            "metrics": metrics or {}
        }
        self._save_registry(self.registry)
        logger.info(f"Registered candidate checkpoint for {round_name} at: {model_path}")

    def promote_checkpoint(self, round_name: str, metrics: Optional[Dict[str, Any]] = None) -> bool:
        if round_name not in self.registry["rounds"]:
            logger.error(f"Cannot promote unknown round: {round_name}")
            return False

        round_data = self.registry["rounds"][round_name]
        round_data["status"] = "promoted"
        if metrics:
            round_data["metrics"] = metrics

        self.registry["active_version"] = round_name
        self.registry["active_path"] = round_data["model_path"]
        self._save_registry(self.registry)
        logger.info(f"PROMOTED CHECKPOINT: {round_name} is now the active policy!")
        return True

    def reject_checkpoint(self, round_name: str, reason: str) -> None:
        if round_name in self.registry["rounds"]:
            self.registry["rounds"][round_name]["status"] = "rejected"
            self.registry["rounds"][round_name]["rejection_reason"] = reason
            self._save_registry(self.registry)
            logger.warning(f"REJECTED candidate {round_name}: {reason}")

    def list_all_rounds(self) -> List[Dict[str, Any]]:
        rounds_list = []
        for name in ROUNDS_ORDER:
            if name in self.registry["rounds"]:
                info = dict(self.registry["rounds"][name])
                info["name"] = name
                rounds_list.append(info)
        return rounds_list
