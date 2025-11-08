#!/bin/bash
set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source params.env if it exists
if [ -f "$PROJECT_ROOT/params.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/params.env" | xargs)
fi

# Default STORAGE_DIR if not set
STORAGE_DIR=${STORAGE_DIR:-/StudentData/reproduce}

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable is not set."
    echo ""
    echo "Please set your Hugging Face token:"
    echo "  export HF_TOKEN='your_token_here'"
    echo ""
    echo "Or add it to params.env file."
    exit 1
fi

# Create models directory in STORAGE_DIR
MODELS_DIR="$STORAGE_DIR/models"
mkdir -p "$MODELS_DIR"

echo "========================================"
echo "Downloading LLM Models"
echo "========================================"
echo "Models directory: $MODELS_DIR"
echo ""

# Model definitions
declare -A MODELS=(
    ["Llama-3.2-3B-Instruct-Q4_K_M.gguf"]="https://huggingface.co/tensorblock/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    ["Llama-3.2-8B-Instruct-Q4_K_M.gguf"]="https://huggingface.co/tensorblock/Llama-3.2-8B-Instruct-GGUF/resolve/main/Llama-3.2-8B-Instruct-Q4_K_M.gguf"
)

# Download each model
for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_URL="${MODELS[$MODEL_NAME]}"
    MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
    
    echo "========================================="
    echo "Model: $MODEL_NAME"
    echo "========================================="
    
    if [ -f "$MODEL_PATH" ]; then
        echo "⏭️  Model already exists: $MODEL_PATH"
        echo "Skipping download..."
        echo ""
    else
        echo "📥 Downloading from: $MODEL_URL"
        echo "📁 Saving to: $MODEL_PATH"
        echo ""
        
        # Download with authorization header
        if wget --header="Authorization: Bearer $HF_TOKEN" \
                --progress=bar:force:noscroll \
                --show-progress \
                "$MODEL_URL" \
                -O "$MODEL_PATH"; then
            echo ""
            echo "✅ Successfully downloaded: $MODEL_NAME"
            
            # Show file size
            FILE_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
            echo "📊 File size: $FILE_SIZE"
            echo ""
        else
            echo ""
            echo "❌ Failed to download: $MODEL_NAME"
            rm -f "$MODEL_PATH"  # Remove partial download
            exit 1
        fi
    fi
done

echo ""
echo "========================================"
echo "✅ All models processed!"
echo "========================================"
echo ""
echo "Models location: $MODELS_DIR"
echo ""
echo "Downloaded models:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "No .gguf files found"
echo ""

