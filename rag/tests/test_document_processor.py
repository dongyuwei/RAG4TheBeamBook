"""Unit tests for document_processor module."""

import tempfile
from pathlib import Path

import pytest

from beam_rag.document_processor import DocumentChunk, DocumentProcessor


class TestDocumentChunk:
    """Test DocumentChunk dataclass."""

    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            content="Test content",
            source="test.txt",
            chunk_type="text",
            section="Introduction",
            metadata={"key": "value"},
        )
        assert chunk.content == "Test content"
        assert chunk.source == "test.txt"
        assert chunk.chunk_type == "text"
        assert chunk.section == "Introduction"
        assert chunk.metadata == {"key": "value"}

    def test_document_chunk_defaults(self):
        """Test DocumentChunk with default values."""
        chunk = DocumentChunk(
            content="Test content",
            source="test.txt",
            chunk_type="code",
        )
        assert chunk.section is None
        assert chunk.metadata is None


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    def test_init_with_default_path(self):
        """Test initialization with default book root path."""
        processor = DocumentProcessor()
        assert processor.book_root is not None
        assert isinstance(processor.book_root, Path)

    def test_init_with_custom_path(self):
        """Test initialization with custom book root path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = DocumentProcessor(book_root=tmpdir)
            assert processor.book_root == Path(tmpdir)
            assert processor.chapters_dir == Path(tmpdir) / "chapters"
            assert processor.code_dir == Path(tmpdir) / "code"

    def test_clean_asciidoc(self):
        """Test asciidoc content cleaning."""
        processor = DocumentProcessor()

        # Test bold formatting removal
        content = "This is *bold* text"
        cleaned = processor._clean_asciidoc(content)
        assert "*bold*" not in cleaned
        assert "bold" in cleaned

        # Test italic formatting removal
        content = "This is _italic_ text"
        cleaned = processor._clean_asciidoc(content)
        assert "_italic_" not in cleaned
        assert "italic" in cleaned

        # Test include directive removal
        content = "Some text\ninclude::file.txt[]\nMore text"
        cleaned = processor._clean_asciidoc(content)
        assert "include::" not in cleaned

    def test_chunk_text_short(self):
        """Test chunking short text."""
        processor = DocumentProcessor()
        text = "Short text"
        chunks = processor._chunk_text(text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_text_long(self):
        """Test chunking long text."""
        processor = DocumentProcessor()
        text = "A" * 5000
        chunks = processor._chunk_text(text, max_chars=2000, overlap=200)
        assert len(chunks) > 1
        # Check that chunks overlap
        if len(chunks) > 1:
            assert len(chunks[0]) > 1800  # Should be close to max_chars

    def test_chunk_code_erlang(self):
        """Test chunking Erlang code."""
        processor = DocumentProcessor()
        code = """
-module(test).
-export([hello/0]).

hello() ->
    world.

 goodbye() ->
    farewell.
"""
        chunks = processor._chunk_code(code, "erlang")
        assert isinstance(chunks, list)

    def test_chunk_code_c(self):
        """Test chunking C code."""
        processor = DocumentProcessor()
        code = """
int main() {
    return 0;
}

void helper() {
    return;
}
"""
        chunks = processor._chunk_code(code, "c")
        assert isinstance(chunks, list)

    def test_process_asciidoc_file(self):
        """Test processing an asciidoc file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            book_root = Path(tmpdir)
            chapters_dir = book_root / "chapters"
            chapters_dir.mkdir()

            # Create a test asciidoc file
            test_file = chapters_dir / "test_chapter.asciidoc"
            test_content = """= Chapter Title

== Section 1

This is the first section.

== Section 2

This is the second section.
"""
            test_file.write_text(test_content)

            processor = DocumentProcessor(book_root=book_root)
            chunks = processor._process_asciidoc(test_file)

            assert len(chunks) > 0
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.chunk_type == "text" for chunk in chunks)

    def test_process_code_file(self):
        """Test processing a code file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            book_root = Path(tmpdir)
            code_dir = book_root / "code"
            code_dir.mkdir()

            # Create a test Python file
            test_file = code_dir / "test.py"
            test_content = """def hello():
    return "world"
"""
            test_file.write_text(test_content)

            processor = DocumentProcessor(book_root=book_root)
            chunks = processor._process_code_file(test_file)

            assert len(chunks) > 0
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.chunk_type == "code" for chunk in chunks)

    def test_get_code_files(self):
        """Test getting code files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            book_root = Path(tmpdir)
            code_dir = book_root / "code"
            code_dir.mkdir()

            # Create test files with different extensions
            (code_dir / "test.erl").write_text("")
            (code_dir / "test.py").write_text("")
            (code_dir / "test.txt").write_text("")  # Should not be included

            processor = DocumentProcessor(book_root=book_root)
            code_files = processor._get_code_files()

            assert len(code_files) == 2
            assert all(
                f.suffix in {".erl", ".hrl", ".c", ".h", ".py", ".java", ".js", ".ts"}
                for f in code_files
            )

    def test_process_all_empty(self):
        """Test processing when no files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = DocumentProcessor(book_root=tmpdir)
            chunks = processor.process_all()
            assert chunks == []
