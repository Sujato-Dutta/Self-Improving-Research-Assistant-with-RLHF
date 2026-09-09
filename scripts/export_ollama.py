#!/usr/bin/env python
"""
Generates an Ollama Modelfile and export script to serve the final promoted RLHF policy locally.
"""
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.rlhf.checkpoint_manager import CheckpointManager
from src.generator.prompt_templates import SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportOllama")


def generate_modelfile():
    manager = CheckpointManager()
    active_version = manager.get_active_version()
    logger.info(f"Generating Ollama Modelfile for active checkpoint: {active_version}...")

    modelfile_content = f"""# Ollama Modelfile for Self-Improving Research Assistant ({active_version})
# Base architecture: Qwen / Qwen2.5 / Qwen3.5 compatible
FROM qwen2.5:0.5b

# Model Hyperparameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Academic Research Assistant System Prompt
SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

# Conversation Template for Grounded Answering
TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""
    modelfile_path = config.base_dir / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    logger.info(f"Written Modelfile to: {modelfile_path}")

    # Write helper shell script
    script_path = config.base_dir / "scripts" / "build_ollama_model.sh"
    sh_content = f"""#!/bin/bash
# Builds and registers the Ollama model
echo "Creating Ollama model: {config.ollama.model_tag}..."
ollama create {config.ollama.model_tag} -f ./Modelfile
echo "Ollama model registered. Test with:"
echo "ollama run {config.ollama.model_tag} 'What is attention mechanism?'"
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(sh_content)
    logger.info(f"Written Ollama build script to: {script_path}")


if __name__ == "__main__":
    generate_modelfile()
