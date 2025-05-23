import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model once
embedder = SentenceTransformer('all-mpnet-base-v2')

# Load and chunk data.txt
# Each chunk is a string of up to chunk_size characters

def load_and_chunk_data(file_path, chunk_size=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = []
    for entry in data:
        for value in entry['content'].values():
            # Split long text into smaller chunks
            for i in range(0, len(value), chunk_size):
                chunk = value[i:i+chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
    return chunks

# Compute embeddings for all chunks

def build_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return chunks, embeddings

# Retrieve top-N relevant chunks for a query

def retrieve(query, chunks, embeddings, top_k=5):
    query_emb = embedder.encode([query], convert_to_numpy=True)[0]
    scores = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices] 