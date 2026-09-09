import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
BENCHMARK_DIR = DATA_DIR / "benchmark"
FAISS_DIR = DATA_DIR / "faiss_index"
DB_PATH = DATA_DIR / "research_assistant.db"

# Ensure essential directories exist
for directory in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, CHECKPOINTS_DIR, BENCHMARK_DIR, FAISS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    # Default model target; fallback-safe and easily overridden via env var
    policy_model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    reference_model_name: str = os.getenv("REF_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reward_model_name: str = os.getenv("REWARD_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    max_seq_length: int = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
    device: str = os.getenv("DEVICE", "cuda" if (HAS_TORCH and torch.cuda.is_available() or os.getenv("CUDA_VISIBLE_DEVICES")) else "cpu")
    torch_dtype: str = "bfloat16" if os.getenv("USE_BF16", "1") == "1" and HAS_TORCH and torch.cuda.is_available() else "float32"


@dataclass
class RetrievalConfig:
    top_k: int = 5
    chunk_size: int = 400
    chunk_overlap: int = 60
    similarity_threshold: float = 0.35
    max_arxiv_results: int = 100
    categories: list = field(default_factory=lambda: ["cs.AI", "cs.CL", "cs.LG", "cs.DC"])


@dataclass
class RewardModelConfig:
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    # Multi-objective weights
    preference_weight: float = 1.0
    relevance_weight: float = 0.35
    citation_correctness_weight: float = 0.40
    groundedness_weight: float = 0.45
    completeness_weight: float = 0.25
    hallucination_penalty: float = 0.50
    verbosity_penalty: float = 0.15


@dataclass
class RLHFConfig:
    learning_rate: float = 1.41e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    ppo_epochs: int = 4
    init_kl_coef: float = 0.15
    target_kl: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    max_grad_norm: float = 0.5
    min_preference_buffer_size: int = 20  # Minimum samples to trigger retraining


@dataclass
class EvaluationGateConfig:
    min_win_rate_vs_base: float = 0.51
    min_reward_delta: float = 0.01
    max_hallucination_rate: float = 0.18
    min_citation_accuracy: float = 0.80
    confidence_level: float = 0.95
    bootstrap_resamples: int = 1000


@dataclass
class OllamaConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_tag: str = os.getenv("OLLAMA_MODEL", "qwen-research-assistant:latest")
    timeout_seconds: float = 30.0


@dataclass
class AppConfig:
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    benchmark_dir: Path = BENCHMARK_DIR
    faiss_dir: Path = FAISS_DIR
    db_url: str = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DATA_DIR}/mlflow.db")
    mlflow_experiment_name: str = "Self-Improving-Research-Assistant-RLHF"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reward: RewardModelConfig = field(default_factory=RewardModelConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)
    gate: EvaluationGateConfig = field(default_factory=EvaluationGateConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


config = AppConfig()
