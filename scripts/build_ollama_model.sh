#!/bin/bash
# Builds and registers the Ollama model
echo "Creating Ollama model: qwen-research-assistant:latest..."
ollama create qwen-research-assistant:latest -f ./Modelfile
echo "Ollama model registered. Test with:"
echo "ollama run qwen-research-assistant:latest 'What is attention mechanism?'"
