import os
import json
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
from src.config import config

logger = logging.getLogger(__name__)

# Fallback seed papers for offline execution and fast testing
SEED_PAPERS = [
    {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
        "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.",
        "url": "https://arxiv.org/abs/1706.03762",
        "published": "2017-06-12",
        "categories": ["cs.CL", "cs.LG"]
    },
    {
        "arxiv_id": "2203.02155",
        "title": "Training language models to follow instructions with human feedback",
        "authors": ["Long Ouyang", "Jeff Wu", "Xu Jiang", "Diogo Almeida", "Carroll L. Wainwright", "Pamela Mishkin", "Chong Zhang", "Sandhini Agarwal", "Katarina Slama", "Alex Ray", "John Schulman", "Jacob Hilton", "Fraser Kelton", "Luke Miller", "Maddie Simens", "Amanda Askell", "Peter Welinder", "Paul Christiano", "Jan Leike", "Ryan Lowe"],
        "summary": "Making language models bigger does not inherently make them better at following a user's intent. Large models can generate outputs that are untruthful, toxic, or simply not helpful. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Using Reinforcement Learning from Human Feedback (RLHF), we train InstructGPT models that users strongly prefer over 100x larger GPT-3 models.",
        "url": "https://arxiv.org/abs/2203.02155",
        "published": "2022-03-04",
        "categories": ["cs.CL", "cs.AI", "cs.LG"]
    },
    {
        "arxiv_id": "2305.18290",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "authors": ["Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon", "Christopher D. Manning", "Chelsea Finn"],
        "summary": "While large language models (LLMs) are capable of impressive generation, controlling their behavior typically requires Reinforcement Learning from Human Feedback (RLHF). Existing RLHF pipelines involve fitting a reward model to preference data, followed by fine-tuning the language model via policy gradients (e.g., PPO). We show that the constrained policy optimization problem can be solved exactly, yielding a simple classification loss directly on preference pairs termed Direct Preference Optimization (DPO).",
        "url": "https://arxiv.org/abs/2305.18290",
        "published": "2023-05-29",
        "categories": ["cs.LG", "cs.AI", "cs.CL"]
    },
    {
        "arxiv_id": "2106.09685",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"],
        "summary": "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to specific tasks. Fine-tuning all parameters becomes prohibitive. We propose Low-Rank Adaptation (LoRA), which freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.",
        "url": "https://arxiv.org/abs/2106.09685",
        "published": "2021-06-17",
        "categories": ["cs.CL", "cs.AI", "cs.LG"]
    },
    {
        "arxiv_id": "2212.08073",
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "authors": ["Yuntao Bai", "Saurav Kadavath", "S warmup", "Amanda Askell", "Jackson Kernion", "Andy Jones", "Anna Chen", "Anna Goldie", "Azalia Mirhoseini", "Cameron Cameron", "Catherine Olsson", "Christopher Clark", "Christopher Lovitt", "Danielle Drain", "Dario Amodei", "Dawn Drain", "Deep Ganguli", "Dustin Li", "Eli Tran-Johnson", "Ethan Perez", "Jamie Kerr", "Jared Mueller", "Jeffrey Ladish", "Joshua Landau", "Kamal Ndousse", "Kamile Lukosuite", "Liane Lovitt", "Michael Sellitto", "Nelson Elhage", "Nicholas Schiefer", "Noemi Mercado", "Nova DasSarma", "Robert Lasenby", "Robin Larson", "Sam Ringer", "Scott Johnston", "Shauna Kravec", "Sheer El Showk", "Stanislav Fort", "Tamera Lanham", "Timothy Telleen-Lawton", "Tom Conerly", "Tom Henighan", "Tristan Hume", "Samuel R. Bowman", "Zac Hatfield-Dodds", "Ben Mann", "Dario Amodei", "Nicholas Joseph", "Sam McCandlish", "Tom Brown", "Christopher Olah"],
        "summary": "As larger and more capable language models are deployed, aligning them with human intentions becomes critical. We propose Constitutional AI, an approach to training harmless AI assistants through self-improvement without human feedback labels for harmfulness. The method uses a list of rules (a constitution) and model self-critique to generate training data for supervised learning and reinforcement learning from AI feedback (RLAIF).",
        "url": "https://arxiv.org/abs/2212.08073",
        "published": "2022-12-15",
        "categories": ["cs.CL", "cs.AI", "cs.LG"]
    },
    {
        "arxiv_id": "2205.14135",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "authors": ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Ré"],
        "summary": "Transformers are slow and memory-hungry on long sequences, as the time and memory complexity of self-attention are quadratic in sequence length. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. FlashAttention trains Transformers faster than existing baselines and enables sequence length scaling.",
        "url": "https://arxiv.org/abs/2205.14135",
        "published": "2022-05-27",
        "categories": ["cs.LG", "cs.AI"]
    },
    {
        "arxiv_id": "2305.14314",
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": ["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"],
        "summary": "We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA), introducing 4-bit NormalFloat data type and Double Quantization.",
        "url": "https://arxiv.org/abs/2305.14314",
        "published": "2023-05-23",
        "categories": ["cs.LG", "cs.AI", "cs.CL"]
    },
    {
        "arxiv_id": "2302.04761",
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "authors": ["Timo Schick", "Jane Dwivedi-Yu", "Roberto Dessì", "Roberta Raileanu", "Maria Lomeli", "Luke Zettlemoyer", "Nicola Cancedda", "Thomas Scialom"],
        "summary": "Language models exhibit remarkable capabilities to solve new tasks from only a few examples or textual instructions. However, they struggle with basic tasks such as arithmetic or factual lookup. We show that LMs can teach themselves to use external tools via simple APIs, achieving the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results.",
        "url": "https://arxiv.org/abs/2302.04761",
        "published": "2023-02-09",
        "categories": ["cs.CL", "cs.AI"]
    },
    {
        "arxiv_id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal", "Heinrich Küttler", "Mike Lewis", "Wen-tau Yih", "Tim Rocktäschel", "Sebastian Riedel", "Douwe Kiela"],
        "summary": "Large pre-trained language models have been shown to store factual knowledge in their parameters. However, their ability to access and precisely manipulate knowledge is still limited. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation, yielding superior groundedness and factual accuracy.",
        "url": "https://arxiv.org/abs/2005.11401",
        "published": "2020-05-22",
        "categories": ["cs.CL", "cs.AI", "cs.IR"]
    },
    {
        "arxiv_id": "2401.06059",
        "title": "A Survey on Retrieval-Augmented Generation for Large Language Models",
        "authors": ["Yunfan Gao", "Yun Xiong", "Xinyu Gao", "Kangxiang Jia", "Jinliu Pan", "Yuxi Bi", "Yi Dai", "Jiawei Sun", "Meng Wang", "Haofen Wang"],
        "summary": "Large Language Models (LLMs) have achieved remarkable success, yet they face challenges like hallucinations, outdated internal knowledge, and lack of domain-specific expertise. Retrieval-Augmented Generation (RAG) mitigates these issues by incorporating external authoritative knowledge bases into the generation process. This paper categorizes RAG into Naive RAG, Advanced RAG, and Modular RAG, analyzing retrieval metrics, citation mechanisms, and generation quality.",
        "url": "https://arxiv.org/abs/2401.06059",
        "published": "2024-01-10",
        "categories": ["cs.CL", "cs.AI", "cs.IR"]
    },
    {
        "arxiv_id": "2005.14165",
        "title": "Language Models are Few-Shot Learners",
        "authors": ["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal", "Arvind Neelakantan", "Pranav Shyam", "Girish Sastry", "Amanda Askell", "Sandhini Agarwal", "Ariel Herbert-Voss", "Gretchen Krueger", "Tom Henighan", "Rewon Child", "Aditya Ramesh", "Daniel M. Ziegler", "Jeffrey Wu", "Clemens Winter", "Christopher Hesse", "Mark Chen", "Eric Sigler", "Mateusz Litwin", "Scott Gray", "Benjamin Chess", "Jack Clark", "Christopher Berner", "Sam McCandlish", "Alec Radford", "Ilya Sutskever", "Dario Amodei"],
        "summary": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. We show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches with GPT-3 175B.",
        "url": "https://arxiv.org/abs/2005.14165",
        "published": "2020-05-28",
        "categories": ["cs.CL"]
    },
    {
        "arxiv_id": "2303.08774",
        "title": "GPT-4 Technical Report",
        "authors": ["OpenAI"],
        "summary": "We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. Alignment through RLHF significantly improves factual accuracy and adherence to steering.",
        "url": "https://arxiv.org/abs/2303.08774",
        "published": "2023-03-15",
        "categories": ["cs.CL", "cs.AI"]
    }
]


class ArxivFetcher:
    """Fetches research papers from arXiv with offline caching and seed paper fallbacks."""

    def __init__(self, cache_file: Path = config.data_dir / "papers_cache.json"):
        self.cache_file = cache_file
        self.papers = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read paper cache: {e}")
        # Initialize with seed papers
        initial = {p["arxiv_id"]: p for p in SEED_PAPERS}
        self._save_cache(initial)
        return initial

    def _save_cache(self, papers: Dict[str, Dict[str, Any]]) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save paper cache: {e}")

    def fetch_papers_online(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Fetch papers from official arXiv API Atom feed."""
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        results = []

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ResearchAssistantRLHF/1.0 (academic; mailto:contact@example.org)"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            for entry in root.findall("atom:entry", ns):
                arxiv_id_elem = entry.find("atom:id", ns)
                raw_id = arxiv_id_elem.text if arxiv_id_elem is not None else ""
                arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

                title_elem = entry.find("atom:title", ns)
                title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else "Untitled"

                summary_elem = entry.find("atom:summary", ns)
                summary = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else ""

                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text)

                published_elem = entry.find("atom:published", ns)
                published = published_elem.text[:10] if published_elem is not None and published_elem.text else ""

                paper_data = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "published": published,
                    "categories": ["cs.AI"]
                }
                results.append(paper_data)
                self.papers[arxiv_id] = paper_data

            self._save_cache(self.papers)
            logger.info(f"Retrieved {len(results)} papers from arXiv for query: '{query}'")
        except Exception as e:
            logger.warning(f"arXiv online fetch error ({e}), falling back to local cached/seed papers.")
            results = self.search_local_papers(query, max_results)

        return results

    def search_local_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Filter local seed and cached papers by keyword matching."""
        query_words = set(query.lower().split())
        scored = []
        for paper in self.papers.values():
            text = (paper["title"] + " " + paper["summary"]).lower()
            score = sum(1 for w in query_words if w in text)
            scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:max_results]]

    def get_all_papers(self) -> List[Dict[str, Any]]:
        return list(self.papers.values())
