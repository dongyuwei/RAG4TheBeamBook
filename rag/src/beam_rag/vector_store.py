"""Vector store using ChromaDB and Sentence Transformers."""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from beam_rag.document_processor import DocumentChunk


def get_cached_model_path(model_name: str) -> str | None:
    """Get the local path for a cached model, or None if not cached.

    Args:
        model_name: The HuggingFace model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')

    Returns:
        Path to the cached model directory, or None if not found
    """
    try:
        # Try to get the model from cache without downloading
        local_path = snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            resume_download=False,
        )
        return local_path
    except Exception:
        # Model not in cache
        return None


class VectorStore:
    """Vector store for BEAM book documents using ChromaDB."""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "beam_book",
    ):
        """Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data.
            embedding_model: Name of the sentence-transformers model.
            collection_name: Name of the ChromaDB collection.
        """
        if persist_directory is None:
            persist_directory = Path(__file__).parent.parent / "vector_db"
        else:
            persist_directory = Path(persist_directory)

        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with filesystem storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "BEAM book knowledge base"},
        )

        # Initialize embedding model - try to use cached version first
        cached_path = get_cached_model_path(embedding_model)
        if cached_path:
            # Use cached model to avoid network calls
            self.embedding_model = SentenceTransformer(
                cached_path, local_files_only=True
            )
        else:
            # Fall back to downloading (will use default cache location)
            self.embedding_model = SentenceTransformer(embedding_model)

    def add_documents(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the vector store.

        Args:
            chunks: List of document chunks to add.

        Returns:
            Number of documents added.
        """
        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        seen_ids = set()

        for chunk in chunks:
            # Generate unique ID based on content hash
            chunk_id = self._generate_id(chunk)

            # Skip if already seen in this batch
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            # Skip if already exists in database
            existing = self.collection.get(ids=[chunk_id])
            if existing and existing["ids"]:
                continue

            ids.append(chunk_id)
            documents.append(chunk.content)

            # Prepare metadata
            metadata = {
                "source": chunk.source,
                "chunk_type": chunk.chunk_type,
                "section": chunk.section or "",
            }
            if chunk.metadata:
                # Convert metadata to strings for ChromaDB
                for key, value in chunk.metadata.items():
                    metadata[key] = str(value)
            metadatas.append(metadata)

            # Generate embedding
            embedding = self.embedding_model.encode(chunk.content)
            embeddings.append(embedding.tolist())

        # Add to collection if there are new documents
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        return len(ids)

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents.

        Args:
            query: The search query.
            n_results: Number of results to return.
            filter_dict: Optional filter for metadata.

        Returns:
            List of search results with content and metadata.
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                formatted_results.append(
                    {
                        "content": doc,
                        "source": metadata.get("source", "unknown"),
                        "chunk_type": metadata.get("chunk_type", "text"),
                        "section": metadata.get("section", ""),
                        "metadata": metadata,
                        "distance": distance,
                        "rank": i + 1,
                    }
                )

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection.name,
            "document_count": count,
            "persist_directory": str(self.persist_directory),
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
        }

    def _generate_id(self, chunk: DocumentChunk) -> str:
        """Generate a unique ID for a document chunk.

        Args:
            chunk: The document chunk.

        Returns:
            Unique ID string.
        """
        # Create ID from content hash and source
        content_hash = hashlib.md5(
            f"{chunk.source}:{chunk.content[:100]}".encode()
        ).hexdigest()
        return content_hash[:16]

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "BEAM book knowledge base"},
        )
