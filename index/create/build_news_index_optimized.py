#!/usr/bin/env python3
"""
Memory-optimized FAISS index builder for large news datasets (30GB+)
Processes data in chunks to avoid memory crashes
"""

import argparse
import json
import os
import re
import uuid
import math
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterator
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import pickle
from tqdm import tqdm
import gc

def clean_text(s: str) -> str:
    """Clean text by normalizing whitespace"""
    s = re.sub(r'\s+', ' ', (s or '')).strip()
    return s

def chunk_text(text: str, target_tokens=700, overlap_tokens=120, enc=None):
    """Token-aware chunking with overlap; returns list of strings."""
    enc = enc or tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + target_tokens)
        chunk = enc.decode(toks[i:j])
        chunks.append(chunk)
        if j == len(toks): 
            break
        i = max(0, j - overlap_tokens)
    return chunks

@dataclass
class ChunkRow:
    doc_id: str
    chunk_id: int
    label: str
    title: str
    source_domain: str
    published_at: str
    url: str
    chunk_text: str

def detect_domain(url: str) -> str:
    """Extract domain from URL"""
    if not url: 
        return ""
    m = re.search(r'://([^/]+)/?', url)
    return m.group(1).lower() if m else ""

def map_news_type_to_label(news_type: str) -> str:
    """Map news type to our label system"""
    if pd.isna(news_type):
        return "other"
    
    news_type = str(news_type).lower()
    
    # Fake/unreliable categories
    if news_type in ['fake', 'unreliable', 'conspiracy', 'clickbait', 'satire', 'hate', 'junksci', 'rumor']:
        return "fake"
    # Reliable categories  
    elif news_type in ['reliable', 'political']:
        return "credible"
    # Everything else
    else:
        return "other"

def stream_csv_chunks(file_path: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
    """Stream CSV file in chunks to avoid loading entire file into memory"""
    print(f"Streaming CSV in chunks of {chunk_size} rows...")
    
    # First, get total number of rows for progress tracking
    total_rows = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_rows += 1
    total_rows -= 1  # Subtract header row
    
    print(f"Total rows to process: {total_rows:,}")
    
    # Now stream the data
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size, iterator=True)
    
    for i, chunk in enumerate(chunk_iter):
        print(f"Processing chunk {i+1} ({len(chunk)} rows)...")
        yield chunk
        
        # Force garbage collection every few chunks
        if i % 10 == 0:
            gc.collect()

def process_chunk_to_chunks(chunk_df: pd.DataFrame, enc: tiktoken.Encoding, 
                           chunk_tokens: int, overlap_tokens: int) -> List[ChunkRow]:
    """Process a dataframe chunk into text chunks"""
    processed_chunks = []
    
    for _, row in chunk_df.iterrows():
        # Skip rows with no content
        if pd.isna(row.get('content')) or len(str(row.get('content', '')).strip()) < 50:
            continue
            
        # Clean and process the row
        doc_id = str(row.get('id', '')) if not pd.isna(row.get('id')) else str(uuid.uuid4())
        text = clean_text(str(row.get('content', '')))
        title = clean_text(str(row.get('title', '')))
        url = str(row.get('url', ''))
        published_at = str(row.get('scraped_at', ''))
        news_type = str(row.get('type', ''))
        
        # Map to our label system
        label = map_news_type_to_label(news_type)
        source_domain = detect_domain(url)
        
        # Chunk the text
        text_chunks = chunk_text(text, target_tokens=chunk_tokens, 
                               overlap_tokens=overlap_tokens, enc=enc)
        
        # Create chunk objects
        for i, chunk_content in enumerate(text_chunks):
            processed_chunks.append(ChunkRow(
                doc_id=doc_id,
                chunk_id=i,
                label=label,
                title=title,
                source_domain=source_domain,
                published_at=published_at,
                url=url,
                chunk_text=chunk_content
            ))
    
    return processed_chunks

def build_embeddings_incrementally(chunks: List[ChunkRow], model: SentenceTransformer, 
                                 batch_size: int = 64) -> np.ndarray:
    """Build embeddings incrementally to manage memory"""
    texts = [c.chunk_text for c in chunks]
    all_embeddings = []
    
    print(f"Encoding {len(texts)} chunks in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts, 
            batch_size=batch_size, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        all_embeddings.append(batch_embeddings.astype("float32"))
        
        # Clear memory periodically
        if i % 100 == 0:
            gc.collect()
    
    return np.vstack(all_embeddings)

def build_news_index_optimized(args):
    """Build FAISS index from large news CSV with memory optimization"""
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Memory-Optimized News Index Builder ===")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.outdir}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Processing limit: {args.limit or 'All data'}")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Initialize embedding model
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Initialize storage
    all_chunks = []
    all_tokenized = []
    label_counts = {}
    total_articles = 0
    
    # Process CSV in streaming chunks
    processed_chunks = 0
    for chunk_df in stream_csv_chunks(args.input, chunk_size=args.chunk_size):
        # Apply limit if specified
        if args.limit and total_articles >= args.limit:
            print(f"Reached limit of {args.limit} articles")
            break
            
        # Process this chunk
        chunk_objects = process_chunk_to_chunks(
            chunk_df, enc, args.chunk_tokens, args.overlap_tokens
        )
        
        # Update counters
        total_articles += len(chunk_df)
        processed_chunks += len(chunk_objects)
        
        # Count labels
        for chunk in chunk_objects:
            label_counts[chunk.label] = label_counts.get(chunk.label, 0) + 1
        
        # Store chunks and tokenized data
        all_chunks.extend(chunk_objects)
        for chunk in chunk_objects:
            toks = re.findall(r"\w+", chunk.chunk_text.lower())
            all_tokenized.append(toks)
        
        print(f"  -> Processed {len(chunk_objects)} text chunks")
        print(f"  -> Total articles so far: {total_articles:,}")
        print(f"  -> Total chunks so far: {processed_chunks:,}")
        
        # Memory management: if we have too many chunks, process them in batches
        if len(all_chunks) >= args.max_chunks_in_memory:
            print(f"Reached memory limit ({args.max_chunks_in_memory} chunks), processing batch...")
            process_batch_and_save(all_chunks, all_tokenized, model, out, args, is_final_batch=False)
            
            # Clear memory
            all_chunks = []
            all_tokenized = []
            gc.collect()
    
    # Process remaining chunks
    if all_chunks:
        print("Processing final batch...")
        process_batch_and_save(all_chunks, all_tokenized, model, out, args, is_final_batch=True)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total articles processed: {total_articles:,}")
    print(f"Total chunks created: {processed_chunks:,}")
    print(f"Label distribution: {label_counts}")

def process_batch_and_save(chunks: List[ChunkRow], tokenized: List[List[str]], 
                          model: SentenceTransformer, out: Path, args, 
                          is_final_batch: bool = False):
    """Process a batch of chunks and save to files"""
    if not chunks:
        return
        
    print(f"Processing batch of {len(chunks)} chunks...")
    
    # Save chunks to JSONL
    if is_final_batch:
        chunks_path = out / "chunks.jsonl"
    else:
        chunks_path = out / f"chunks_batch_{len(chunks)}.jsonl"
    
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    
    # Build embeddings for this batch
    embeddings = build_embeddings_incrementally(chunks, model, batch_size=args.embed_batch_size)
    
    # Save FAISS index
    if is_final_batch:
        index_path = out / "faiss.index"
    else:
        index_path = out / f"faiss_batch_{len(chunks)}.index"
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    
    # Save BM25
    if is_final_batch:
        bm25_path = out / "bm25.pkl"
    else:
        bm25_path = out / f"bm25_batch_{len(chunks)}.pkl"
    
    bm25 = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks_meta": [asdict(c) for c in chunks]}, f)
    
    print(f"Saved batch: {len(chunks)} chunks, {embeddings.shape[0]} embeddings")
    
    # Create metadata for final batch
    if is_final_batch:
        meta = {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "dim": int(embeddings.shape[1]),
            "num_chunks": int(embeddings.shape[0]),
            "bm25_docs": int(len(tokenized)),
            "chunk_tokens": args.chunk_tokens,
            "overlap_tokens": args.overlap_tokens,
            "source_file": str(args.input)
        }
        
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata → {out / 'meta.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Memory-optimized FAISS index builder for large news datasets")
    ap.add_argument("--input", required=True, help="Path to news CSV file")
    ap.add_argument("--outdir", default="index/store", help="Output directory for index files")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of articles to process")
    ap.add_argument("--chunk-size", type=int, default=500, help="CSV chunk size for streaming")
    ap.add_argument("--chunk-tokens", type=int, default=700, help="Target tokens per text chunk")
    ap.add_argument("--overlap-tokens", type=int, default=120, help="Overlap tokens between chunks")
    ap.add_argument("--max-chunks-in-memory", type=int, default=50000, help="Max chunks to keep in memory")
    ap.add_argument("--embed-batch-size", type=int, default=64, help="Batch size for embedding generation")
    
    args = ap.parse_args()
    build_news_index_optimized(args)
