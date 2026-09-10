import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from src.config import config

logger = logging.getLogger(__name__)

BENCHMARK_TOPICS = {
    "llm_architectures": [
        "How does the self-attention mechanism in the Transformer eliminate recurrent dependencies?",
        "Compare multi-head attention versus grouped-query attention (GQA) in inference memory consumption.",
        "What are the computational tradeoffs of FlashAttention tiling versus standard softmax materialization?",
        "How does Rotary Position Embedding (RoPE) encode relative token distances in Qwen and LLaMA?",
        "Explain the KV-cache memory bottleneck during long-context autoregressive token generation.",
        "How does sliding-window attention maintain bounded memory during extended sequence evaluation?",
        "Discuss the stability properties of Pre-LayerNorm vs Post-LayerNorm in deep Transformer training.",
        "What is the mathematical formulation of cross-attention in encoder-decoder architectures?",
        "How does mixture-of-experts (MoE) achieve sparse computation without sacrificing total model capacity?",
        "Explain how tokenization granularity (BPE vs WordPiece) impacts out-of-vocabulary representation."
    ],
    "alignment_and_rlhf": [
        "What are the mathematical differences between PPO and Direct Preference Optimization (DPO)?",
        "How does the Bradley-Terry preference model formulate pairwise probabilities in reward modeling?",
        "Why is KL divergence regularization strictly required when fine-tuning policy models with PPO?",
        "How does Constitutional AI implement automated self-critique without human harmfulness labels?",
        "Discuss the failure mode of reward hacking and length bias in reinforcement learning from human feedback.",
        "How does Kahneman-Tversky Optimization (KTO) align models using unpaired binary signals?",
        "Explain the role of Generalized Advantage Estimation (GAE) in reducing policy gradient variance.",
        "What is the impact of reference model freeze vs periodic EMA updates in actor-critic alignment?",
        "How does rejection sampling fine-tuning (Best-of-N) compare to online RLHF policy optimization?",
        "Discuss calibration evaluation metrics (Brier score, ECE) for learned language model reward models."
    ],
    "rag_and_retrieval": [
        "What are the key trade-offs between dense semantic retrieval (FAISS) and sparse lexical search (BM25)?",
        "How does inner product similarity on normalized vectors correspond to cosine distance in FAISS?",
        "Explain the mechanism of Hypothetical Document Embeddings (HyDE) in zero-shot retrieval.",
        "How does contextual chunk overlap prevent semantic truncation across document boundaries?",
        "What are the standard failure modes when retrieving evidence for multi-hop reasoning questions?",
        "How can cross-encoder rerankers improve precision over dual-encoder bi-encoders in RAG?",
        "Discuss the groundedness verification protocols needed to prevent hallucinated citations in LLMs.",
        "How does parent-document retrieval balance small chunk granularity with large context windows?",
        "Explain the difference between Inverted File (IVF) and Hierarchical Navigable Small World (HNSW) indexing.",
        "What metrics are best suited for evaluating evidence extraction recall versus generation faithfulness?"
    ],
    "peft_and_quantization": [
        "How does Low-Rank Adaptation (LoRA) decompose weight updates without increasing inference latency?",
        "What is the mathematical advantage of 4-bit NormalFloat (NF4) quantization in QLoRA?",
        "Explain how double quantization in QLoRA saves additional GPU memory across quantization constants.",
        "How does Activation-aware Weight Quantization (AWQ) protect salient outlier channels in LLMs?",
        "Compare post-training quantization (PTQ) vs quantization-aware training (QAT) for Transformer weights.",
        "What causes catastrophic activation spikes in large scale LLMs and how does SmoothQuant address them?",
        "Explain how LoRA ranks (r) and scaling alpha influence gradient stability during fine-tuning.",
        "How does FlashDecoding parallelize along the sequence length dimension during the decoding phase?",
        "Discuss the quantization format of GGUF and its optimization for CPU and Ollama deployment.",
        "What are the memory bandwidth bottlenecks of INT4 matrix multiplications on tensor cores?"
    ],
    "reasoning_and_agents": [
        "How does Chain-of-Thought (CoT) prompting alter the token probability distribution during generation?",
        "Explain the synergy between reasoning traces and external action execution in the ReAct paradigm.",
        "How does Toolformer autonomously decide when and how to invoke external API tools?",
        "What is the principle of Self-Refine and iterative verbal reinforcement in reasoning tasks?",
        "Discuss Tree-of-Thoughts (ToT) search algorithms (BFS/DFS) for combinatorial problem solving.",
        "How does Reflexion utilize episodic memory buffers to prevent repetitive reasoning errors?",
        "What are the failure modes of autonomous agents when processing unstructured multi-tool schemas?",
        "Explain the role of verifier models (outcome vs process supervision) in mathematical reasoning.",
        "How does prompt decomposition break complex technical synthesis into verified sub-queries?",
        "Discuss the vulnerability of agent tool-calling pipelines to prompt injection and context poisoning."
    ],
    "hpc_and_distributed": [
        "How does SLURM allocate Multi-Instance GPU (MIG) slices on NVIDIA DGX A100 systems?",
        "Explain the difference between ZeRO-1, ZeRO-2, and ZeRO-3 memory partitioning in DeepSpeed.",
        "What are the communication volume differences between AllReduce and ReduceScatter operations?",
        "How does pipeline parallelism handle pipeline bubbles with 1F1B (one forward, one backward) scheduling?",
        "Discuss the impact of NVLink bandwidth versus PCIe Gen4 bandwidth on distributed tensor parallelism.",
        "How does Megatron-LM split row-parallel and column-parallel linear projections in multi-head attention?",
        "What are the best practices for SLURM batch job submission using Enroot and Docker containers?",
        "Explain the gradient accumulation mechanism for simulating large effective batch sizes on limited VRAM.",
        "How does mixed precision training (AMP with bfloat16) prevent underflow while halving memory footprint?",
        "What causes CUDA out-of-memory (OOM) errors during the backward pass despite sufficient forward memory?"
    ],
    "scientific_synthesis": [
        "How do self-improving post-training loops establish statistical confidence in model capability gains?",
        "Discuss non-parametric bootstrap resampling for computing 95% confidence intervals on win rates.",
        "How can automated regression test suites gate model deployment in continuous integration pipelines?",
        "What statistical methods differentiate genuine policy improvements from stochastic generation variance?",
        "Explain the Bradley-Terry preference loss formulation and its relationship to cross-entropy."
    ]
}


def generate_benchmark_suite(target_size: int = 350) -> List[Dict[str, Any]]:
    """Generates a structured held-out benchmark suite of 350 technical questions."""
    benchmark = []
    base_pool = []
    for category, questions in BENCHMARK_TOPICS.items():
        for q in questions:
            base_pool.append({"category": category, "question": q})

    # Expand systematically to exactly target_size using specialized domain variants
    modifiers = [
        "",
        " Provide mathematical foundations and algorithmic details.",
        " Explain the underlying theoretical principles and empirical tradeoffs.",
        " Contrast the state-of-the-art methodology with classical baseline approaches.",
        " Discuss practical implementation considerations and computational bottlenecks."
    ]

    count = 1
    while len(benchmark) < target_size:
        for item in base_pool:
            mod_idx = (count // len(base_pool)) % len(modifiers)
            mod = modifiers[mod_idx]
            q_text = item["question"].rstrip("?") + mod + ("?" if not mod.endswith(".") else "")

            benchmark.append({
                "id": f"bench_{count:04d}",
                "category": item["category"],
                "question": q_text,
                "domain": item["category"].replace("_", " ").title()
            })
            count += 1
            if len(benchmark) >= target_size:
                break

    return benchmark


class BenchmarkManager:
    """Manages loading and saving the 350-question held-out benchmark."""

    def __init__(self, file_path: Path = config.benchmark_dir / "benchmark_350.json"):
        self.file_path = file_path
        self.questions = self._load_or_create()

    def _load_or_create(self) -> List[Dict[str, Any]]:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if len(data) >= 300:
                        logger.info(f"Loaded held-out benchmark with {len(data)} questions.")
                        return data
            except Exception as e:
                logger.warning(f"Failed to load benchmark file: {e}")

        # Create fresh benchmark suite
        suite = generate_benchmark_suite(target_size=350)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(suite, f, indent=2)
        logger.info(f"Generated fresh held-out benchmark of {len(suite)} questions at: {self.file_path}")
        return suite

    def get_questions(self, limit: int = 350) -> List[Dict[str, Any]]:
        return self.questions[:limit]

    def get_benchmark_questions(self, limit: int = 350) -> List[Dict[str, Any]]:
        return self.get_questions(limit=limit)
