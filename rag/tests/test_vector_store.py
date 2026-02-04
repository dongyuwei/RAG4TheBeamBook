"""Unit tests for vector_store module."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beam_rag.document_processor import DocumentChunk
from beam_rag.vector_store import VectorStore, get_cached_model_path


class TestGetCachedModelPath:
    """Test get_cached_model_path function."""

    @patch("beam_rag.vector_store.snapshot_download")
    def test_get_cached_model_path_success(self, mock_snapshot_download):
        """Test getting cached model path when model exists."""
        mock_snapshot_download.return_value = "/path/to/cached/model"
        result = get_cached_model_path("test-model")
        assert result == "/path/to/cached/model"
        mock_snapshot_download.assert_called_once_with(
            repo_id="test-model",
            local_files_only=True,
            resume_download=False,
        )

    @patch("beam_rag.vector_store.snapshot_download")
    def test_get_cached_model_path_not_found(self, mock_snapshot_download):
        """Test getting cached model path when model doesn't exist."""
        mock_snapshot_download.side_effect = Exception("Model not found")
        result = get_cached_model_path("nonexistent-model")
        assert result is None


class TestVectorStore:
    """Test VectorStore class."""

    @patch("beam_rag.vector_store.SentenceTransformer")
    @patch("beam_rag.vector_store.chromadb.PersistentClient")
    def test_init_default(self, mock_client_class, mock_transformer_class):
        """Test VectorStore initialization with default parameters."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_transformer = MagicMock()
        mock_transformer_class.return_value = mock_transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "beam_rag.vector_store.get_cached_model_path", return_value=None
            ):
                store = VectorStore(persist_directory=tmpdir)
                assert store.persist_directory == Path(tmpdir)
                mock_client_class.assert_called_once()
                mock_client.get_or_create_collection.assert_called_once()

    @patch("beam_rag.vector_store.SentenceTransformer")
    @patch("beam_rag.vector_store.chromadb.PersistentClient")
    def test_init_with_cached_model(self, mock_client_class, mock_transformer_class):
        """Test VectorStore initialization with cached model."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_transformer = MagicMock()
        mock_transformer_class.return_value = mock_transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "beam_rag.vector_store.get_cached_model_path",
                return_value="/cached/path",
            ):
                store = VectorStore(persist_directory=tmpdir)
                mock_transformer_class.assert_called_once_with(
                    "/cached/path", local_files_only=True
                )

    @patch("beam_rag.vector_store.SentenceTransformer")
    @patch("beam_rag.vector_store.chromadb.PersistentClient")
    def test_generate_id(self, mock_client_class, mock_transformer_class):
        """Test ID generation for document chunks."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_transformer_class.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "beam_rag.vector_store.get_cached_model_path", return_value=None
            ):
                store = VectorStore(persist_directory=tmpdir)
                chunk = DocumentChunk(
                    content="Test content for hashing",
                    source="test.txt",
                    chunk_type="text",
                )
                chunk_id = store._generate_id(chunk)

                # Verify ID is generated from hash
                expected_hash = hashlib.md5(
                    f"{chunk.source}:{chunk.content[:100]}".encode()
                ).hexdigest()[:16]
                assert chunk_id == expected_hash

    @patch("beam_rag.vector_store.SentenceTransformer")
    @patch("beam_rag.vector_store.chromadb.PersistentClient")
    def test_add_documents_empty(self, mock_client_class, mock_transformer_class):
        """Test adding empty list of documents."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_transformer_class.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "beam_rag.vector_store.get_cached_model_path", return_value=None
            ):
                store = VectorStore(persist_directory=tmpdir)
                result = store.add_documents([])
                assert result == 0

    @patch("beam_rag.vector_store.SentenceTransformer")
    @patch("beam_rag.vector_store.chromadb.PersistentClient")
    def test_add_documents_with_duplicates(
        self, mock_client_class, mock_transformer_class
    ):
        """Test adding documents with duplicates."""
        import numpy as np

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}  # No existing docs
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_transformer = MagicMock()
        mock_embedding = np.array([0.1] * 384)
        mock_transformer.encode.return_value = mock_embedding
        mock_transformer_class.return_value = mock_transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "beam_rag.vector_store.get_cached_model_path", return_value=None
            ):
                store = VectorStore(persist_directory=tmpdir)

                # Create two identical chunks
                chunk = DocumentChunk(
                    content="Duplicate content",
                    source="test.txt",
                    chunk_type="text",
                )
                chunks = [chunk, chunk]

                result = store.add_documents(chunks)
                # Should only add one because duplicates are skipped
                assert result == 1

    @patch("beam_rag.vector_store.SentenceTransformer")
    @patch("beam_rag.vector_store.chromadb.PersistentClient")
    def test_search(self, mock_client_class, mock_transformer_class):
        """Test searching documents."""
        import numpy as np

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        # Mock search results
        mock_collection.query.return_value = {
            "documents": [["Test document content"]],
            "metadatas": [
                [{"source": "test.txt", "chunk_type": "text", "section": "Intro"}]
            ],
            "distances": [[0.5]],
        }

        mock_transformer = MagicMock()
        mock_embedding = np.array([0.1] * 384)
        mock_transformer.encode.return_value = mock_embedding
        mock_transformer_class.return_value = mock_transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "beam_rag.vector_store.get_cached_model_path", return_value=None
            ):
                store = VectorStore(persist_directory=tmpdir)
                results = store.search("test query", n_results=5)

                assert len(results) == 1
                assert results[0]["content"] == "Test document content"
                assert results[0]["source"] == "test.txt"
                assert results[0]["rank"] == 1

    @patch("beam_rag.vector_store.SentenceTransformer")
    @patch("beam_rag.vector_store.chromadb.PersistentClient")
    def test_search_no_results(self, mock_client_class, mock_transformer_class):
        """Test searching when no results found."""
        import numpy as np

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        # Mock empty results
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_transformer = MagicMock()
        mock_embedding = np.array([0.1] * 384)
        mock_transformer.encode.return_value = mock_embedding
        mock_transformer_class.return_value = mock_transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "beam_rag.vector_store.get_cached_model_path", return_value=None
            ):
                store = VectorStore(persist_directory=tmpdir)
                results = store.search("test query")

                assert results == []

    @patch("beam_rag.vector_store.SentenceTransformer")
    @patch("beam_rag.vector_store.chromadb.PersistentClient")
    def test_get_collection_stats(self, mock_client_class, mock_transformer_class):
        """Test getting collection statistics."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.name = "test_collection"
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_transformer = MagicMock()
        mock_transformer.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "beam_rag.vector_store.get_cached_model_path", return_value=None
            ):
                store = VectorStore(persist_directory=tmpdir)
                stats = store.get_collection_stats()

                assert stats["collection_name"] == "test_collection"
                assert stats["document_count"] == 100
                assert stats["embedding_model"] == 384
                assert stats["persist_directory"] == tmpdir

    @patch("beam_rag.vector_store.SentenceTransformer")
    @patch("beam_rag.vector_store.chromadb.PersistentClient")
    def test_clear(self, mock_client_class, mock_transformer_class):
        """Test clearing the collection."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_transformer_class.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "beam_rag.vector_store.get_cached_model_path", return_value=None
            ):
                store = VectorStore(persist_directory=tmpdir)
                store.clear()

                mock_client.delete_collection.assert_called_once_with("test_collection")
                # Should recreate the collection
                assert mock_client.get_or_create_collection.call_count == 2
