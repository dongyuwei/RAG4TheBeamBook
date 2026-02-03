"""Document processor for BEAM book asciidoc and code files."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DocumentChunk:
    """A chunk of document content with metadata."""

    content: str
    source: str
    chunk_type: str  # 'text' or 'code'
    section: str | None = None
    metadata: dict | None = None


class DocumentProcessor:
    """Process asciidoc and code files for the BEAM book."""

    def __init__(self, book_root: Path | str | None = None):
        """Initialize processor with path to the book root directory.

        Args:
            book_root: Path to the BEAM book repository root.
                        Defaults to parent of rag directory.
        """
        if book_root is None:
            # Default: assume we're in rag/ folder, book root is parent
            self.book_root = Path(__file__).parent.parent.parent.parent
        else:
            self.book_root = Path(book_root)

        self.chapters_dir = self.book_root / "chapters"
        self.code_dir = self.book_root / "code"

    def process_all(self) -> List[DocumentChunk]:
        """Process all asciidoc and code files.

        Returns:
            List of document chunks with metadata.
        """
        chunks = []

        # Process asciidoc files from chapters/
        if self.chapters_dir.exists():
            for asciidoc_file in self.chapters_dir.glob("*.asciidoc"):
                chunks.extend(self._process_asciidoc(asciidoc_file))

        # Process code files from code/
        if self.code_dir.exists():
            for code_file in self._get_code_files():
                chunks.extend(self._process_code_file(code_file))

        return chunks

    def _get_code_files(self) -> List[Path]:
        """Get all code files from the code directory."""
        code_files = []
        code_extensions = {".erl", ".hrl", ".c", ".h", ".py", ".java", ".js", ".ts"}

        if self.code_dir.exists():
            for ext in code_extensions:
                code_files.extend(self.code_dir.rglob(f"*{ext}"))

        return code_files

    def _process_asciidoc(self, filepath: Path) -> List[DocumentChunk]:
        """Process an asciidoc file and extract chunks.

        Args:
            filepath: Path to the asciidoc file.

        Returns:
            List of document chunks from the file.
        """
        chunks = []
        content = filepath.read_text(encoding="utf-8")

        # Get chapter name from filename
        chapter_name = filepath.stem

        # Split by sections (== Section Name)
        sections = re.split(r"\n(?=={1,4}[^=])", content)

        for section_content in sections:
            if not section_content.strip():
                continue

            # Extract section title if present
            section_match = re.match(r"={1,4}\s*(.+?)(?:\n|$)", section_content)
            section_title = section_match.group(1) if section_match else None

            # Remove asciidoc formatting for cleaner text
            clean_content = self._clean_asciidoc(section_content)

            # Create chunks (split long sections into smaller chunks)
            text_chunks = self._chunk_text(clean_content, max_chars=2000)

            for chunk_text in text_chunks:
                if chunk_text.strip():
                    chunks.append(
                        DocumentChunk(
                            content=chunk_text,
                            source=str(filepath.relative_to(self.book_root)),
                            chunk_type="text",
                            section=section_title,
                            metadata={"chapter": chapter_name},
                        )
                    )

        return chunks

    def _process_code_file(self, filepath: Path) -> List[DocumentChunk]:
        """Process a code file and extract chunks.

        Args:
            filepath: Path to the code file.

        Returns:
            List of document chunks from the file.
        """
        chunks = []
        content = filepath.read_text(encoding="utf-8")

        # Get language from extension
        ext = filepath.suffix
        language_map = {
            ".erl": "erlang",
            ".hrl": "erlang",
            ".c": "c",
            ".h": "c",
            ".py": "python",
            ".java": "java",
            ".js": "javascript",
            ".ts": "typescript",
        }
        language = language_map.get(ext, "code")

        # Split code into meaningful chunks (functions, classes, etc.)
        code_chunks = self._chunk_code(content, language)

        for chunk_text in code_chunks:
            if chunk_text.strip():
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        source=str(filepath.relative_to(self.book_root)),
                        chunk_type="code",
                        section=None,
                        metadata={
                            "language": language,
                            "filename": filepath.name,
                        },
                    )
                )

        return chunks

    def _clean_asciidoc(self, content: str) -> str:
        """Remove asciidoc formatting markers from content."""
        # Remove include directives
        content = re.sub(r"^include::[^\[]+\[\]", "", content, flags=re.MULTILINE)

        # Remove xref references but keep the text
        content = re.sub(r"xref:([^\[]+)\[([^\]]*)\]", r"\2", content)

        # Remove link directives but keep the text
        content = re.sub(r"link:[^\[]*\[([^\]]+)\]", r"\1", content)

        # Remove index markers ((...))
        content = re.sub(r"\(\([^)]+\)\)", "", content)

        # Remove block markers
        content = re.sub(r"^\[.*?\]$", "", content, flags=re.MULTILINE)

        # Remove inline formatting (*bold*, _italic_)
        content = re.sub(r"\*([^*]+)\*", r"\1", content)
        content = re.sub(r"_([^_]+)_", r"\1", content)

        # Remove code block delimiters
        content = re.sub(r"^----$", "", content, flags=re.MULTILINE)

        # Remove asciidoc headers (lines starting with =)
        content = re.sub(r"^=+\s*", "", content, flags=re.MULTILINE)

        # Clean up extra whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()

    def _chunk_text(
        self, text: str, max_chars: int = 2000, overlap: int = 200
    ) -> List[str]:
        """Split text into chunks with overlap.

        Args:
            text: The text to split.
            max_chars: Maximum characters per chunk.
            overlap: Number of characters to overlap between chunks.

        Returns:
            List of text chunks.
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_chars

            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for sentence boundary (. followed by space or newline)
                sentence_break = text.rfind(". ", start, end)
                if sentence_break > start + max_chars // 2:
                    end = sentence_break + 1
                else:
                    # Look for paragraph break
                    para_break = text.rfind("\n\n", start, end)
                    if para_break > start + max_chars // 2:
                        end = para_break + 2

            chunks.append(text[start:end].strip())
            start = end - overlap

        return chunks

    def _chunk_code(self, content: str, language: str) -> List[str]:
        """Split code into meaningful chunks.

        Args:
            content: The code content.
            language: The programming language.

        Returns:
            List of code chunks.
        """
        chunks = []

        if language == "erlang":
            # Split Erlang code by function definitions
            # Pattern: function_name(...) -> ... end.
            pattern = (
                r"([a-z][a-zA-Z0-9_]*\s*\([^)]*\)\s*(?:when\s+[^->]+)?\s*->.*?\n\.)"
            )
            matches = re.findall(pattern, content, re.DOTALL)
            chunks.extend(matches)
        elif language == "c":
            # Split C code by function definitions
            # Pattern: return_type function_name(...) { ... }
            pattern = r"([a-zA-Z_][a-zA-Z0-9_\s*]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*\{[^}]*\})"
            matches = re.findall(pattern, content, re.DOTALL)
            chunks.extend(matches)
        else:
            # For other languages, split by larger blocks or use fixed size
            chunks = self._chunk_text(content, max_chars=3000, overlap=100)

        # If no specific patterns matched, use the whole file
        if not chunks and content.strip():
            chunks = [content]

        return chunks
