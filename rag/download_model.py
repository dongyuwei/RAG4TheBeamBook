#!/usr/bin/env python3
"""Download and cache the sentence-transformers model."""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Downloading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print(f"✓ Model downloaded successfully!")
print(f"  Model: {MODEL_NAME}")
print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
print(f"  Max sequence length: {model.max_seq_length}")
