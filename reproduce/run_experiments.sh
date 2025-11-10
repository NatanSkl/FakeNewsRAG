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
RAG2_PYTHON=${RAG2_PYTHON:-$STORAGE_DIR/rag2/bin/python}
LLM_URL=${LLM_URL:-http://127.0.0.1:8010}
STORE_PATH=${STORE_PATH:-$STORAGE_DIR/index}

# Directories
PREPROCESSED_DIR="$STORAGE_DIR/preprocessed"
EXPERIMENTS_DIR="$STORAGE_DIR/experiments"

echo "========================================"
echo "Running Experiments"
echo "========================================"
echo "Storage directory: $STORAGE_DIR"
echo "RAG2 Python: $RAG2_PYTHON"
echo "DEBUG_REPRODUCE: $DEBUG_REPRODUCE"
echo "LLM URL: $LLM_URL"
echo "Store path: $STORE_PATH"
echo ""

# Check if RAG2 Python exists
if [ ! -f "$RAG2_PYTHON" ]; then
    echo "❌ Error: RAG2 Python not found at: $RAG2_PYTHON"
    echo "Please run reproduce/create_conda_envs.sh first"
    exit 1
fi

# Create experiments directory
mkdir -p "$EXPERIMENTS_DIR"

# Determine limit flag based on DEBUG_REPRODUCE
LIMIT_ARGS=()
if [ "$DEBUG_REPRODUCE" = "1" ]; then
    LIMIT_ARGS=("--limit" "2")
    echo "🐛 DEBUG MODE: Limiting to 2 articles per experiment"
    echo "   Limit args: ${LIMIT_ARGS[@]}"
fi

# ==============================================================================
# Step 1: Sample validation and test sets
# ==============================================================================

echo ""
echo "========================================="
echo "Step 1: Sampling validation and test sets"
echo "========================================="

# Sample validation set
VAL_INPUT="$PREPROCESSED_DIR/val.csv"
VAL_OUTPUT="$PREPROCESSED_DIR/val_sampled.csv"

if [ ! -f "$VAL_INPUT" ]; then
    echo "❌ Error: Validation file not found: $VAL_INPUT"
    exit 1
fi

echo "Sampling 100 rows from validation set..."
$RAG2_PYTHON "$PROJECT_ROOT/evaluate/sample.py" \
    "$VAL_INPUT" \
    100 \
    "$VAL_OUTPUT"

if [ ! -f "$VAL_OUTPUT" ]; then
    echo "❌ Error: Failed to create sampled validation file"
    exit 1
fi
echo "✅ Created: $VAL_OUTPUT"
echo ""

# Sample test set
TEST_INPUT="$PREPROCESSED_DIR/test.csv"
TEST_OUTPUT="$PREPROCESSED_DIR/test_sampled.csv"

if [ ! -f "$TEST_INPUT" ]; then
    echo "❌ Error: Test file not found: $TEST_INPUT"
    exit 1
fi

echo "Sampling 100 rows from test set..."
$RAG2_PYTHON "$PROJECT_ROOT/evaluate/sample.py" \
    "$TEST_INPUT" \
    100 \
    "$TEST_OUTPUT"

if [ ! -f "$TEST_OUTPUT" ]; then
    echo "❌ Error: Failed to create sampled test file"
    exit 1
fi
echo "✅ Created: $TEST_OUTPUT"
echo ""

# ==============================================================================
# Step 2: Run experiments on validation set
# ==============================================================================

echo ""
echo "========================================="
echo "Step 2: Running experiments on validation set"
echo "========================================="

# Experiment 1: default optimized, prompt-type 0, fake_reliable
echo ""
echo "Experiment 1: default optimized, prompt-type 0, fake_reliable"
echo "------------------------------------------------------------"
echo "$RAG2_PYTHON \"$PROJECT_ROOT/evaluate/experiment_runner.py\" \\"
echo "    \"$VAL_OUTPUT\" \\"
echo "    --retrieval-configs default optimized \\"
echo "    --prompt-types 0 \\"
echo "    --naming-conventions fake_reliable \\"
echo "    --store-path \"$STORE_PATH\" \\"
echo "    --llm-url \"$LLM_URL\" \\"
echo "    --output-dir \"$EXPERIMENTS_DIR\" \\"
echo "    \"${LIMIT_ARGS[@]}\""
$RAG2_PYTHON "$PROJECT_ROOT/evaluate/experiment_runner.py" \
    "$VAL_OUTPUT" \
    --retrieval-configs default optimized \
    --prompt-types 0 \
    --naming-conventions fake_reliable \
    --store-path "$STORE_PATH" \
    --llm-url "$LLM_URL" \
    --output-dir "$EXPERIMENTS_DIR" \
    "${LIMIT_ARGS[@]}"

# Experiment 2: optimized, prompt-types 0 1 2, naming fake_reliable type1_type2
echo ""
echo "Experiment 2: optimized, prompt-types 0 1 2, naming fake_reliable type1_type2"
echo "------------------------------------------------------------"
$RAG2_PYTHON "$PROJECT_ROOT/evaluate/experiment_runner.py" \
    "$VAL_OUTPUT" \
    --retrieval-configs optimized \
    --prompt-types 0 1 2 \
    --naming-conventions fake_reliable type1_type2 \
    --store-path "$STORE_PATH" \
    --llm-url "$LLM_URL" \
    --output-dir "$EXPERIMENTS_DIR" \
    "${LIMIT_ARGS[@]}"

# ==============================================================================
# Step 3: Run experiments on test set
# ==============================================================================

echo ""
echo "========================================="
echo "Step 3: Running experiments on test set"
echo "========================================="

# Experiment 3: prompt-type 2, naming fake_reliable on test set
echo ""
echo "Experiment 3: prompt-type 2, naming fake_reliable (test set)"
echo "------------------------------------------------------------"
$RAG2_PYTHON "$PROJECT_ROOT/evaluate/experiment_runner.py" \
    "$TEST_OUTPUT" \
    --retrieval-configs optimized \
    --prompt-types 2 \
    --naming-conventions fake_reliable \
    --store-path "$STORE_PATH" \
    --llm-url "$LLM_URL" \
    --output-dir "$EXPERIMENTS_DIR" \
    "${LIMIT_ARGS[@]}"

# Experiment 4: Only LLM executor on test set
echo ""
echo "Experiment 4: Only LLM executor (test set)"
echo "------------------------------------------------------------"
$RAG2_PYTHON "$PROJECT_ROOT/evaluate/only_llm_executor.py" \
    "$TEST_OUTPUT" \
    --llm-url "$LLM_URL" \
    --output-dir "$EXPERIMENTS_DIR" \
    "${LIMIT_ARGS[@]}"

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "========================================"
echo "✅ All experiments completed!"
echo "========================================"
echo ""
echo "Results saved to: $EXPERIMENTS_DIR"
echo ""
echo "Experiments run:"
echo "  1. Validation: default optimized, prompt-type 0, fake_reliable"
echo "  2. Validation: optimized, prompt-types 0 1 2, naming fake_reliable type1_type2"
echo "  3. Test: prompt-type 2, naming fake_reliable"
echo "  4. Test: Only LLM executor"
if [ "$DEBUG_REPRODUCE" = "1" ]; then
    echo ""
    echo "⚠️  DEBUG MODE: All experiments limited to 2 articles"
fi
echo ""

