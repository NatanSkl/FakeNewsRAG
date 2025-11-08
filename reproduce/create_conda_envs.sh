#!/bin/bash
set -e  # Exit on error

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source params.env if it exists
if [ -f "$PROJECT_ROOT/params.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/params.env" | xargs)
fi

# Default STORAGE_DIR if not set
STORAGE_DIR=${STORAGE_DIR:-/StudentData/reproduce}

echo "========================================"
echo "Creating conda environments"
echo "Storage directory: $STORAGE_DIR"
echo "========================================"

# Create storage directory if it doesn't exist
mkdir -p "$STORAGE_DIR"

# Set up cache and temporary directories to use storage directory
# This prevents filling up the regular disk
export CONDA_PKGS_DIRS="$STORAGE_DIR/conda_pkgs"
export PIP_CACHE_DIR="$STORAGE_DIR/pip_cache"
export TMPDIR="$STORAGE_DIR/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

# ==============================================================================
# Environment 1: llama-cuda
# ==============================================================================

ENV_LLAMA_PATH="$STORAGE_DIR/llama-cuda"

echo ""
echo "========================================="
echo "Environment 1: llama-cuda"
echo "========================================="

if [ -d "$ENV_LLAMA_PATH" ]; then
    echo "⏭️  llama-cuda environment already exists at: $ENV_LLAMA_PATH"
    echo "Skipping creation..."
else
    echo ""
    echo "Step 1/5: Creating fresh conda environment with Python 3.11..."
    conda create -p "$ENV_LLAMA_PATH" python=3.11 -y

    echo ""
    echo "Step 2/5: Installing NVCC 11.8 compiler..."
    conda install -p "$ENV_LLAMA_PATH" -y -c nvidia/label/cuda-11.8.0 cuda-nvcc=11.8.89

    echo ""
    echo "Step 3/5: Installing CUDA toolkit 11.8 and build tools..."
    conda remove -p "$ENV_LLAMA_PATH" -y cudatoolkit 2>/dev/null || true
    conda install -p "$ENV_LLAMA_PATH" -y -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8.0
    conda install -p "$ENV_LLAMA_PATH" -c conda-forge cmake ninja git -y

    echo ""
    echo "Step 4/5: Setting up CUDA environment variables..."
    # These will be used for the build
    export CUDACXX="$ENV_LLAMA_PATH/bin/nvcc"
    export CUDA_HOME="$ENV_LLAMA_PATH"
    export CUDA_PATH="$ENV_LLAMA_PATH"
    export CUDA_TOOLKIT_ROOT_DIR="$ENV_LLAMA_PATH"
    export PATH="$ENV_LLAMA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$ENV_LLAMA_PATH/lib:$LD_LIBRARY_PATH"

    echo "Checking NVCC version..."
    "$ENV_LLAMA_PATH/bin/nvcc" --version

    echo ""
    echo "Step 5/5: Building llama-cpp-python 0.3.16 with CUDA support (this may take several minutes)..."
    export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=52 -DCUDAToolkit_ROOT=$ENV_LLAMA_PATH"
    export FORCE_CMAKE=1

    # Use the environment's pip
    "$ENV_LLAMA_PATH/bin/pip" install --no-cache-dir --upgrade pip
    "$ENV_LLAMA_PATH/bin/pip" install --no-cache-dir --upgrade --force-reinstall llama-cpp-python==0.3.16
    
    echo ""
    echo "Installing additional requirements from llama_cuda_requirements.txt..."
    # Install other packages from requirements file (excluding llama-cpp-python which we already built)
    grep -v "^llama_cpp_python" "$PROJECT_ROOT/requirements/llama_cuda_requirements.txt" | \
        grep -v "^$" | \
        xargs "$ENV_LLAMA_PATH/bin/pip" install --no-cache-dir

    echo ""
    echo "✅ llama-cuda environment created successfully!"
    
    # Test the installation
    echo "Testing llama-cpp-python installation..."
    "$ENV_LLAMA_PATH/bin/python" -c "import llama_cpp; print(f'llama-cpp-python version: {llama_cpp.__version__}')" && echo "✅ Import successful!" || echo "❌ Import failed!"
fi

# ==============================================================================
# Environment 2: rag2
# ==============================================================================

ENV_RAG2_PATH="$STORAGE_DIR/rag2"

echo ""
echo "========================================="
echo "Environment 2: rag2"
echo "========================================="

if [ -d "$ENV_RAG2_PATH" ]; then
    echo "⏭️  rag2 environment already exists at: $ENV_RAG2_PATH"
    echo "Skipping creation..."
else
    echo ""
    echo "Step 1/4: Creating fresh conda environment with Python 3.10..."
    conda create -p "$ENV_RAG2_PATH" python=3.10 -y

    echo ""
    echo "Step 2/4: Installing cudatoolkit via conda..."
    # Install cudatoolkit via conda
    conda install -p "$ENV_RAG2_PATH" -c pytorch -c nvidia cudatoolkit=11.8 -y

    echo ""
    echo "Step 3/4: Installing requirements from rag2_requirements.txt..."
    "$ENV_RAG2_PATH/bin/pip" install --no-cache-dir --upgrade pip
    
    # Install PyTorch first from PyTorch repository (it has special CUDA versions)
    echo "Installing PyTorch with CUDA 12.1 support..."
    "$ENV_RAG2_PATH/bin/pip" install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    
    # Install faiss-gpu via pip (conda version doesn't support Python 3.10)
    echo "Installing faiss-gpu 1.7.2 via pip..."
    "$ENV_RAG2_PATH/bin/pip" install --no-cache-dir faiss-gpu==1.7.2
    
    # Install remaining requirements, skipping lines with @ file:/// (conda-specific paths) and torch lines
    echo "Installing remaining requirements..."
    FILTERED_REQ_FILE="$STORAGE_DIR/tmp/rag2_requirements_filtered.txt"
    grep -v "@ file:///" "$PROJECT_ROOT/requirements/rag2_requirements.txt" | \
        grep -v "^torch==" | \
        grep -v "^torchvision==" | \
        grep -v "^torchaudio==" | \
        grep -v "^cudatoolkit" | \
        grep -v "^nvidia$" | \
        grep -v "^faiss-gpu" > "$FILTERED_REQ_FILE"
    "$ENV_RAG2_PATH/bin/pip" install --no-cache-dir -r "$FILTERED_REQ_FILE"
    rm "$FILTERED_REQ_FILE"

    echo ""
    echo "Step 4/4: Verifying installation..."
    "$ENV_RAG2_PATH/bin/python" -c "import torch; print(f'PyTorch version: {torch.__version__}')" && echo "✅ PyTorch import successful!" || echo "⚠️  PyTorch import failed"
    "$ENV_RAG2_PATH/bin/python" -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')" && echo "✅ Streamlit import successful!" || echo "⚠️  Streamlit import failed"
    "$ENV_RAG2_PATH/bin/python" -c "import sentence_transformers; print(f'Sentence-transformers version: {sentence_transformers.__version__}')" && echo "✅ Sentence-transformers import successful!" || echo "⚠️  Sentence-transformers import failed"
    
    echo ""
    echo "✅ rag2 environment created successfully!"
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "========================================"
echo "✅ All environments processed!"
echo "========================================"
echo ""
echo "Environments created in: $STORAGE_DIR"
echo ""
echo "1. llama-cuda environment:"
echo "   Location: $ENV_LLAMA_PATH"
echo "   Activate: conda activate $ENV_LLAMA_PATH"
echo "   Direct:   $ENV_LLAMA_PATH/bin/python"
echo ""
echo "2. rag2 environment:"
echo "   Location: $ENV_RAG2_PATH"
echo "   Activate: conda activate $ENV_RAG2_PATH"
echo "   Direct:   $ENV_RAG2_PATH/bin/python"
echo ""
