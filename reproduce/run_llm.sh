#!/bin/bash

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source params.env if it exists
if [ -f "$PROJECT_ROOT/params.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/params.env" | xargs)
fi

# Default STORAGE_DIR if not set
STORAGE_DIR=${STORAGE_DIR:-/StudentData/reproduce}

# Parse argument
MODEL_SIZE=${1:-3B}

# Validate model size
if [ "$MODEL_SIZE" != "3B" ] && [ "$MODEL_SIZE" != "8B" ]; then
    echo "❌ Error: Model size must be either '3B' or '8B'"
    echo "Usage: $0 [3B|8B]"
    exit 1
fi

# Set up paths
ENV_LLAMA_PATH="$STORAGE_DIR/llama-cuda"
MODELS_DIR="$STORAGE_DIR/models"
MODEL_NAME="Llama-3.2-${MODEL_SIZE}-Instruct-Q4_K_M.gguf"
MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
PORT=8010

echo "========================================"
echo "Running Llama ${MODEL_SIZE} Server"
echo "========================================"
echo "Environment: $ENV_LLAMA_PATH"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo ""

# Check if environment exists
if [ ! -d "$ENV_LLAMA_PATH" ]; then
    echo "❌ Error: llama-cuda environment not found at: $ENV_LLAMA_PATH"
    echo "Please run reproduce/create_conda_envs.sh first"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at: $MODEL_PATH"
    echo "Please run reproduce/download_llms.sh first"
    exit 1
fi

# Set CUDA environment variables (from run_llms.txt)
export CUDACXX="$ENV_LLAMA_PATH/bin/nvcc"
export CUDA_HOME="$ENV_LLAMA_PATH"
export CUDA_PATH="$ENV_LLAMA_PATH"
export CUDA_TOOLKIT_ROOT_DIR="$ENV_LLAMA_PATH"
export PATH="$ENV_LLAMA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_LLAMA_PATH/lib:$LD_LIBRARY_PATH"

echo "Starting llama server..."
echo "Server will be available at: http://127.0.0.1:$PORT"
echo "Press Ctrl+C to stop"
echo ""

# Run the server
"$ENV_LLAMA_PATH/bin/python" -m llama_cpp.server \
  --model "$MODEL_PATH" \
  --n_gpu_layers -1 \
  --n_ctx 4096 \
  --n_batch 256 \
  --offload_kqv true \
  --host 127.0.0.1 \
  --port "$PORT"

