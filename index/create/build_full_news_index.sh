#!/bin/bash

# Production script to build FAISS index for the full 30GB news dataset
# This script is designed to handle the large dataset without crashing your laptop

echo "=== Building Full News Index ==="
echo "This will process the full 30GB news dataset"
echo "Estimated time: Several hours"
echo "Memory usage: Controlled to prevent crashes"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run this from the project root."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Create output directory
mkdir -p index/store

echo "Starting index build with optimized settings..."
echo "Settings:"
echo "  - CSV chunk size: 200 (processes 200 rows at a time)"
echo "  - Max chunks in memory: 10000 (prevents memory overflow)"
echo "  - Embedding batch size: 32 (conservative for memory)"
echo "  - Text chunk size: 700 tokens"
echo "  - Overlap: 120 tokens"
echo ""

# Run the optimized index builder
python build_news_index_optimized.py \
    --input ../news_parts/news_cleaned_2018_02_13.csv \
    --outdir ../index/store \
    --chunk-size 200 \
    --chunk-tokens 700 \
    --overlap-tokens 120 \
    --max-chunks-in-memory 10000 \
    --embed-batch-size 32

echo ""
echo "=== Index Build Complete ==="
echo "Batch files created in index/store/"
echo ""
echo "To merge all batches into a single index, run:"
echo "python merge_batch_indices.py --batch-dir index/store --output-dir index/final"
echo ""
echo "To test the index with your RAG pipeline, update the store_dir parameter to point to index/store"
