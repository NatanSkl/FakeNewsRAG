#!/usr/bin/env python3
"""
Merge multiple batch FAISS indices into a single index
"""

import argparse
import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def merge_faiss_indices(batch_files: list, output_path: str):
    """Merge multiple FAISS indices into one"""
    print(f"Merging {len(batch_files)} FAISS indices...")
    
    # Load first index to get dimensions
    first_index = faiss.read_index(str(batch_files[0]))
    dim = first_index.dntotal
    
    # Create new index
    merged_index = faiss.IndexFlatIP(dim)
    
    # Add all vectors from batch indices
    for batch_file in tqdm(batch_files, desc="Merging indices"):
        batch_index = faiss.read_index(str(batch_file))
        vectors = batch_index.reconstruct_n(0, batch_index.ntotal)
        merged_index.add(vectors)
    
    # Save merged index
    faiss.write_index(merged_index, output_path)
    print(f"Saved merged FAISS index: {output_path}")
    print(f"Total vectors: {merged_index.ntotal}")

def merge_bm25_indices(batch_files: list, output_path: str):
    """Merge multiple BM25 indices"""
    print(f"Merging {len(batch_files)} BM25 indices...")
    
    all_tokenized = []
    all_chunks_meta = []
    
    for batch_file in tqdm(batch_files, desc="Loading BM25 batches"):
        with open(batch_file, "rb") as f:
            data = pickle.load(f)
            all_tokenized.extend(data["chunks_meta"])
            # Re-tokenize for BM25
            for chunk in data["chunks_meta"]:
                import re
                toks = re.findall(r"\w+", chunk["chunk_text"].lower())
                all_tokenized.append(toks)
    
    # Create merged BM25
    merged_bm25 = BM25Okapi(all_tokenized)
    
    # Save merged BM25
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": merged_bm25, "chunks_meta": all_chunks_meta}, f)
    
    print(f"Saved merged BM25: {output_path}")

def merge_chunk_files(batch_files: list, output_path: str):
    """Merge multiple chunk JSONL files"""
    print(f"Merging {len(batch_files)} chunk files...")
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        for batch_file in tqdm(batch_files, desc="Merging chunks"):
            with open(batch_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
    
    print(f"Saved merged chunks: {output_path}")

def main():
    ap = argparse.ArgumentParser(description="Merge batch indices into single index")
    ap.add_argument("--batch-dir", required=True, help="Directory containing batch files")
    ap.add_argument("--output-dir", default="index/store", help="Output directory")
    
    args = ap.parse_args()
    
    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all batch files
    faiss_batches = sorted(batch_dir.glob("faiss_batch_*.index"))
    bm25_batches = sorted(batch_dir.glob("bm25_batch_*.pkl"))
    chunk_batches = sorted(batch_dir.glob("chunks_batch_*.jsonl"))
    
    print(f"Found {len(faiss_batches)} FAISS batches")
    print(f"Found {len(bm25_batches)} BM25 batches") 
    print(f"Found {len(chunk_batches)} chunk batches")
    
    if not faiss_batches:
        print("No batch files found!")
        return
    
    # Merge indices
    merge_faiss_indices(faiss_batches, str(output_dir / "faiss.index"))
    merge_bm25_indices(bm25_batches, str(output_dir / "bm25.pkl"))
    merge_chunk_files(chunk_batches, str(output_dir / "chunks.jsonl"))
    
    # Create metadata
    meta = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "num_batches": len(faiss_batches),
        "source": "merged_batches"
    }
    
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print("Merge complete!")

if __name__ == "__main__":
    main()
