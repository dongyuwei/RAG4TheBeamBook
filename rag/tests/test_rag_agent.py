"""Unit tests for rag_agent module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from beam_rag.rag_agent import RAGResponse, RAGAgent


class TestRAGResponse:
    """Test RAGResponse model."""

    def test_rag_response_creation(self):
        """Test creating a RAGResponse."""
        response = RAGResponse(
            answer="Test answer",
            source_snippets=["source1", "source2"],
            confidence="high",
        )
        assert response.answer == "Test answer"
        assert response.source_snippets == ["source1", "source2"]
        assert response.confidence == "high"

    def test_rag_response_validation(self):
        """Test RAGResponse field validation."""
        response = RAGResponse(
            answer="Another answer",
            source_snippets=[],
            confidence="medium",
        )
        assert response.confidence in ["high", "medium", "low"]


class TestRAGAgent:
    """Test RAGAgent class."""

    @patch("beam_rag.rag_agent.VectorStore")
    @patch("beam_rag.rag_agent.Agent")
    def test_init_default(self, mock_agent_class, mock_vector_store_class):
        """Test RAGAgent initialization with default parameters."""
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        agent = RAGAgent()
        assert agent.vector_store is not None
        mock_agent_class.assert_called_once()

    @patch("beam_rag.rag_agent.VectorStore")
    @patch("beam_rag.rag_agent.Agent")
    def test_init_with_custom_params(self, mock_agent_class, mock_vector_store_class):
        """Test RAGAgent initialization with custom parameters."""
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        with patch.dict("os.environ", {}, clear=True):
            agent = RAGAgent(
                vector_store=mock_vector_store,
                model="openai:gpt-4",
                api_key="test-key",
            )
            assert agent.vector_store == mock_vector_store
            mock_agent_class.assert_called_once()

    @patch("beam_rag.rag_agent.VectorStore")
    @patch("beam_rag.rag_agent.Agent")
    def test_init_with_anthropic(self, mock_agent_class, mock_vector_store_class):
        """Test RAGAgent initialization with Anthropic model."""
        mock_vector_store = MagicMock()
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        with patch.dict("os.environ", {}, clear=True):
            agent = RAGAgent(
                vector_store=mock_vector_store,
                model="anthropic:claude-3-opus",
                api_key="anthropic-key",
            )
            mock_agent_class.assert_called_once()

    @pytest.mark.asyncio
    @patch("beam_rag.rag_agent.VectorStore")
    @patch("beam_rag.rag_agent.Agent")
    async def test_ask_no_context(self, mock_agent_class, mock_vector_store_class):
        """Test asking question with no context found."""
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []
        mock_vector_store_class.return_value = mock_vector_store
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        agent = RAGAgent(vector_store=mock_vector_store)
        response = await agent.ask("test question")

        assert "couldn't find any relevant information" in response.answer
        assert response.source_snippets == []
        assert response.confidence == "low"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    @patch("beam_rag.rag_agent.VectorStore")
    @patch("beam_rag.rag_agent.Agent")
    async def test_ask_with_multiple_results(
        self, mock_agent_class, mock_vector_store_class
    ):
        """Test asking question with multiple context results."""
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {
                "content": "Content 1",
                "source": "file1.txt",
                "section": "Section 1",
                "chunk_type": "text",
            },
            {
                "content": "Content 2",
                "source": "file2.txt",
                "section": None,
                "chunk_type": "code",
            },
        ]
        mock_vector_store_class.return_value = mock_vector_store

        mock_result = MagicMock()
        mock_result.output = RAGResponse(
            answer="Answer based on multiple sources",
            source_snippets=[],
            confidence="medium",
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent_class.return_value = mock_agent

        agent = RAGAgent(vector_store=mock_vector_store)
        response = await agent.ask("test question", n_results=2)

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with("test question", n_results=2)

    @patch("beam_rag.rag_agent.VectorStore")
    @patch("beam_rag.rag_agent.Agent")
    def test_build_context(self, mock_agent_class, mock_vector_store_class):
        """Test building context string from results."""
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_agent_class.return_value = MagicMock()

        agent = RAGAgent(vector_store=mock_vector_store)
        results = [
            {
                "content": "Test content",
                "source": "test.txt",
                "section": "Introduction",
                "chunk_type": "text",
            }
        ]
        context = agent._build_context(results)

        assert "Document 1" in context
        assert "test.txt" in context
        assert "Introduction" in context
        assert "Test content" in context

    @patch("beam_rag.rag_agent.VectorStore")
    @patch("beam_rag.rag_agent.Agent")
    def test_build_context_no_section(self, mock_agent_class, mock_vector_store_class):
        """Test building context when result has no section."""
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_agent_class.return_value = MagicMock()

        agent = RAGAgent(vector_store=mock_vector_store)
        results = [
            {
                "content": "Code content",
                "source": "code.erl",
                "section": None,
                "chunk_type": "code",
            }
        ]
        context = agent._build_context(results)

        assert "code.erl" in context
        assert "Section:" not in context  # Should not include section line

    @patch("beam_rag.rag_agent.VectorStore")
    @patch("beam_rag.rag_agent.Agent")
    def test_get_stats(self, mock_agent_class, mock_vector_store_class):
        """Test getting statistics."""
        mock_vector_store = MagicMock()
        mock_vector_store.get_collection_stats.return_value = {
            "document_count": 50,
            "collection_name": "test",
        }
        mock_vector_store_class.return_value = mock_vector_store
        mock_agent_class.return_value = MagicMock()

        agent = RAGAgent(vector_store=mock_vector_store)
        stats = agent.get_stats()

        assert stats["document_count"] == 50
        mock_vector_store.get_collection_stats.assert_called_once()
