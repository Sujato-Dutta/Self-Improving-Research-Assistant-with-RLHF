import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from src.config import config
from src.generator.policy_model import PolicyGenerator, extract_citations
from src.generator.prompt_templates import SYSTEM_PROMPT, build_research_prompt

logger = logging.getLogger(__name__)


class OllamaAdapter:
    """Ollama local serving adapter with automatic fallback to local PyTorch/HF PolicyGenerator."""

    def __init__(
        self,
        base_url: str = config.ollama.base_url,
        model_tag: str = config.ollama.model_tag,
        fallback_generator: Optional[PolicyGenerator] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.model_tag = model_tag
        self.fallback = fallback_generator or PolicyGenerator()

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2.0) as response:
                return response.status == 200
        except Exception:
            return False

    def generate(self, query: str, evidence: list, temperature: float = 0.3) -> Dict[str, Any]:
        prompt = build_research_prompt(query, evidence)
        if self.is_available():
            try:
                payload = {
                    "model": self.model_tag,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=config.ollama.timeout_seconds) as response:
                    res_json = json.loads(response.read().decode("utf-8"))
                    text = res_json.get("response", "")
                    citations = extract_citations(text)
                    return {
                        "text": text,
                        "citations": citations,
                        "source": "ollama",
                        "model": self.model_tag
                    }
            except Exception as e:
                logger.warning(f"Ollama generation request failed ({e}), falling back to local generator.")

        # Local fallback
        text, citations, latency = self.fallback.generate_single(query, evidence, temperature=temperature)
        return {
            "text": text,
            "citations": citations,
            "source": "local_policy",
            "latency_ms": latency,
            "model": config.model.policy_model_name
        }
