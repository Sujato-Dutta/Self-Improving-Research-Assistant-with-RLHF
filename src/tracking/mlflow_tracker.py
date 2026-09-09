import os
import json
import logging
from typing import Dict, Any, Optional
from src.config import config

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks metrics, hyperparameters, and model checkpoints using MLflow."""

    def __init__(
        self,
        tracking_uri: str = config.mlflow_tracking_uri,
        experiment_name: str = config.mlflow_experiment_name
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow = None
        self._active_run = None
        self._init_mlflow()

    def _init_mlflow(self):
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow initialized with tracking URI: {self.tracking_uri}")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow ({e}). Logging locally.")
            self._mlflow = None

    def start_run(self, run_name: str) -> Any:
        if self._mlflow is not None:
            try:
                self._active_run = self._mlflow.start_run(run_name=run_name)
                return self._active_run
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")
        return None

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._mlflow is not None:
            try:
                # Convert complex objects to strings
                clean_params = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v for k, v in params.items()}
                self._mlflow.log_params(clean_params)
            except Exception as e:
                logger.warning(f"MLflow log_params failed: {e}")
        logger.info(f"Parameters: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._mlflow is not None:
            try:
                numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                self._mlflow.log_metrics(numeric_metrics, step=step)
            except Exception as e:
                logger.warning(f"MLflow log_metrics failed: {e}")
        logger.info(f"Metrics (step={step}): {metrics}")

    def log_artifact(self, file_path: str) -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.log_artifact(file_path)
            except Exception as e:
                logger.warning(f"MLflow log_artifact failed: {e}")

    def end_run(self) -> None:
        if self._mlflow is not None and self._active_run is not None:
            try:
                self._mlflow.end_run()
                self._active_run = None
            except Exception as e:
                logger.warning(f"MLflow end_run failed: {e}")
