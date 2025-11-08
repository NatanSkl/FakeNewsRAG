#!/bin/bash
set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source params.env if it exists
if [ -f "$PROJECT_ROOT/params.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/params.env" | xargs)
fi

# Default values
STORAGE_DIR=${STORAGE_DIR:-/StudentData/reproduce}
DEBUG_REPRODUCE=${DEBUG_REPRODUCE:-0}
RAG2_PYTHON=${RAG2_PYTHON:-/StudentData/reproduce/rag2/bin/python}

# Directories
DATA_DIR="$STORAGE_DIR/data"
PREPROCESSED_DIR="$STORAGE_DIR/preprocessed"
INDEX_FAKE_DIR="$STORAGE_DIR/index_fake"
INDEX_RELIABLE_DIR="$STORAGE_DIR/index_reliable"

echo "========================================"
echo "Get Files & Build Index Pipeline"
echo "========================================"
echo "Storage directory: $STORAGE_DIR"
echo "RAG2 Python: $RAG2_PYTHON"
echo "DEBUG_REPRODUCE: $DEBUG_REPRODUCE"
echo ""

# Check if RAG2 Python exists
if [ ! -f "$RAG2_PYTHON" ]; then
    echo "❌ Error: RAG2 Python not found at: $RAG2_PYTHON"
    echo "Please run reproduce/create_conda_envs.sh first"
    exit 1
fi

# ==============================================================================
# Step 1: Download CSV files
# ==============================================================================

echo "========================================="
echo "Step 1: Downloading CSV files"
echo "========================================="

# Set STORAGE_DIR for get_csv.py
export STORAGE_DIR="$STORAGE_DIR"

cd "$PROJECT_ROOT/index"

# Run get_csv.py (it will download and extract)
echo "Running get_csv.py..."
echo "Downloading to: $DATA_DIR"
$RAG2_PYTHON get_csv.py

# Verify the file exists
if [ -f "$DATA_DIR/news.csv" ]; then
    FILE_SIZE=$(du -h "$DATA_DIR/news.csv" | cut -f1)
    echo "✅ CSV file ready at: $DATA_DIR/news.csv ($FILE_SIZE)"
else
    echo "❌ Error: news.csv not found at $DATA_DIR/news.csv"
    exit 1
fi

echo ""

# ==============================================================================
# Step 2: Preprocess CSV files
# ==============================================================================

echo "========================================="
echo "Step 2: Preprocessing CSV files"
echo "========================================="

mkdir -p "$PREPROCESSED_DIR"

# Check if preprocessed files already exist
if [ -f "$PREPROCESSED_DIR/train_fake.csv" ] && [ -f "$PREPROCESSED_DIR/train_reliable.csv" ]; then
    echo "⏭️  Preprocessed files already exist, skipping preprocessing..."
    FAKE_SIZE=$(du -h "$PREPROCESSED_DIR/train_fake.csv" | cut -f1)
    RELIABLE_SIZE=$(du -h "$PREPROCESSED_DIR/train_reliable.csv" | cut -f1)
    echo "✅ Preprocessed files ready:"
    echo "   - train_fake.csv ($FAKE_SIZE)"
    echo "   - train_reliable.csv ($RELIABLE_SIZE)"
else
    echo "Running preprocess_csv.py..."
    $RAG2_PYTHON preprocess_csv.py \
        --input "$DATA_DIR/news.csv" \
        --out-dir "$PREPROCESSED_DIR"

    if [ -f "$PREPROCESSED_DIR/train_fake.csv" ] && [ -f "$PREPROCESSED_DIR/train_reliable.csv" ]; then
        echo "✅ Preprocessing complete!"
        echo "   - train_fake.csv"
        echo "   - train_reliable.csv"
    else
        echo "❌ Error: Preprocessing failed - output files not found"
        exit 1
    fi
fi

echo ""

# ==============================================================================
# Step 3: Build FAISS indices
# ==============================================================================

echo "========================================="
echo "Step 3: Building FAISS indices"
echo "========================================="

# Determine limit flag based on DEBUG_REPRODUCE
LIMIT_FLAG=""
if [ "$DEBUG_REPRODUCE" = "1" ]; then
    LIMIT_FLAG="--limit 20000"
    echo "🐛 DEBUG MODE: Limiting to 20,000 rows"
fi

# Build fake index
echo ""
echo "Building FAKE index..."
echo "Input: $PREPROCESSED_DIR/train_fake.csv"
echo "Output: $INDEX_FAKE_DIR"
echo ""

$RAG2_PYTHON "$PROJECT_ROOT/index/build_index_v3.py" \
    --input "$PREPROCESSED_DIR/train_fake.csv" \
    --out-dir "$INDEX_FAKE_DIR" \
    --normalize \
    --use-encoding \
    --batch-size 256 \
    --append \
    --chunk-tokens 128 \
    --save-metadata-as csv \
    --chunk-size 20000 \
    $LIMIT_FLAG

echo ""
echo "✅ FAKE index built successfully!"
echo ""

# Build reliable index
echo "Building RELIABLE index..."
echo "Input: $PREPROCESSED_DIR/train_reliable.csv"
echo "Output: $INDEX_RELIABLE_DIR"
echo ""

$RAG2_PYTHON "$PROJECT_ROOT/index/build_index_v3.py" \
    --input "$PREPROCESSED_DIR/train_reliable.csv" \
    --out-dir "$INDEX_RELIABLE_DIR" \
    --normalize \
    --use-encoding \
    --batch-size 256 \
    --append \
    --chunk-tokens 128 \
    --save-metadata-as csv \
    --chunk-size 20000 \
    $LIMIT_FLAG

echo ""
echo "✅ RELIABLE index built successfully!"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "========================================"
echo "✅ Pipeline Complete!"
echo "========================================"
echo ""
echo "Data directory:        $DATA_DIR"
echo "Preprocessed directory: $PREPROCESSED_DIR"
echo "Fake index:            $INDEX_FAKE_DIR"
echo "Reliable index:        $INDEX_RELIABLE_DIR"
echo ""
echo "Index files:"
ls -lh "$INDEX_FAKE_DIR"/*.index 2>/dev/null || echo "  (no .index files in $INDEX_FAKE_DIR)"
ls -lh "$INDEX_RELIABLE_DIR"/*.index 2>/dev/null || echo "  (no .index files in $INDEX_RELIABLE_DIR)"
echo ""

