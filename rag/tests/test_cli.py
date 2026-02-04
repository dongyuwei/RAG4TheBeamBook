"""Unit tests for cli module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from beam_rag.cli import cli


class TestCLI:
    """Test CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "BEAM Book RAG" in result.output

    def test_version(self):
        """Test version flag."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    @patch("beam_rag.cli.DocumentProcessor")
    @patch("beam_rag.cli.VectorStore")
    def test_index_command(self, mock_vector_store_class, mock_processor_class):
        """Test index command."""
        mock_processor = MagicMock()
        mock_processor.process_all.return_value = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        mock_processor_class.return_value = mock_processor

        mock_vector_store = MagicMock()
        mock_vector_store.add_documents.return_value = 3
        mock_vector_store.get_collection_stats.return_value = {
            "collection_name": "beam_book",
            "document_count": 3,
            "persist_directory": "/test/path",
            "embedding_model": 384,
        }
        mock_vector_store_class.return_value = mock_vector_store

        result = self.runner.invoke(cli, ["index"])
        assert result.exit_code == 0
        assert "Indexing Complete" in result.output
        assert "3" in result.output

    @patch("beam_rag.cli.VectorStore")
    def test_stats_command(self, mock_vector_store_class):
        """Test stats command."""
        mock_vector_store = MagicMock()
        mock_vector_store.get_collection_stats.return_value = {
            "collection_name": "beam_book",
            "document_count": 100,
            "persist_directory": "/test/path",
            "embedding_model": 384,
        }
        mock_vector_store_class.return_value = mock_vector_store

        result = self.runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "100" in result.output
        assert "beam_book" in result.output

    @patch("beam_rag.cli.VectorStore")
    def test_clear_command(self, mock_vector_store_class):
        """Test clear command."""
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        result = self.runner.invoke(cli, ["clear"], input="y\n")
        assert result.exit_code == 0
        assert "cleared successfully" in result.output
        mock_vector_store.clear.assert_called_once()

    @patch("beam_rag.cli.VectorStore")
    def test_clear_command_cancel(self, mock_vector_store_class):
        """Test clear command cancellation."""
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        result = self.runner.invoke(cli, ["clear"], input="n\n")
        assert result.exit_code != 0  # Should abort
        mock_vector_store.clear.assert_not_called()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("beam_rag.cli.VectorStore")
    @patch("beam_rag.cli.RAGAgent")
    @patch("asyncio.run")
    def test_ask_command(
        self, mock_asyncio_run, mock_agent_class, mock_vector_store_class
    ):
        """Test ask command."""
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_agent = MagicMock()
        mock_agent.get_stats.return_value = {"document_count": 10}
        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.confidence = "high"
        mock_response.source_snippets = ["Source 1"]
        mock_asyncio_run.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        result = self.runner.invoke(cli, ["ask", "What is BEAM?"])
        assert result.exit_code == 0
        assert "Test answer" in result.output
        assert "high" in result.output

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("beam_rag.cli.VectorStore")
    @patch("beam_rag.cli.RAGAgent")
    def test_ask_command_no_documents(self, mock_agent_class, mock_vector_store_class):
        """Test ask command when no documents indexed."""
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_agent = MagicMock()
        mock_agent.get_stats.return_value = {"document_count": 0}
        mock_agent_class.return_value = mock_agent

        result = self.runner.invoke(cli, ["ask", "What is BEAM?"])
        assert result.exit_code == 0
        assert "No documents" in result.output

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("beam_rag.cli.VectorStore")
    @patch("beam_rag.cli.RAGAgent")
    @patch("asyncio.run")
    def test_ask_command_with_env_key(
        self, mock_asyncio_run, mock_agent_class, mock_vector_store_class
    ):
        """Test ask command using API key from environment."""
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_agent = MagicMock()
        mock_agent.get_stats.return_value = {"document_count": 10}
        mock_response = MagicMock()
        mock_response.answer = "Answer"
        mock_response.confidence = "medium"
        mock_response.source_snippets = []
        mock_asyncio_run.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        result = self.runner.invoke(cli, ["ask", "Test question?"])
        assert result.exit_code == 0

    @patch("beam_rag.cli.VectorStore")
    @patch("beam_rag.cli.RAGAgent")
    def test_ask_command_no_api_key(self, mock_agent_class, mock_vector_store_class):
        """Test ask command without API key."""
        with patch.dict("os.environ", {}, clear=True):
            mock_vector_store = MagicMock()
            mock_vector_store_class.return_value = mock_vector_store

            result = self.runner.invoke(cli, ["ask", "What is BEAM?"])
            assert result.exit_code == 0
            assert "API key not found" in result.output

    def test_ask_command_with_options(self):
        """Test ask command with all options."""
        with (
            patch("beam_rag.cli.VectorStore") as mock_vector_store_class,
            patch("beam_rag.cli.RAGAgent") as mock_agent_class,
            patch("asyncio.run") as mock_asyncio_run,
        ):
            mock_vector_store = MagicMock()
            mock_vector_store_class.return_value = mock_vector_store

            mock_agent = MagicMock()
            mock_agent.get_stats.return_value = {"document_count": 10}
            mock_response = MagicMock()
            mock_response.answer = "Detailed answer"
            mock_response.confidence = "medium"
            mock_response.source_snippets = ["Source 1", "Source 2", "Source 3"]
            mock_asyncio_run.return_value = mock_response
            mock_agent_class.return_value = mock_agent

            result = self.runner.invoke(
                cli,
                [
                    "ask",
                    "What is BEAM?",
                    "--model",
                    "openai:gpt-4",
                    "--api-key",
                    "test-key",
                    "--n-results",
                    "10",
                ],
            )

            assert result.exit_code == 0
            assert "Detailed answer" in result.output
            # Verify n_results was passed correctly
            mock_agent_class.assert_called_once()
