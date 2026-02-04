# BEAM Book RAG

A Personal Knowledge Retriever for the Erlang BEAM book using Pydantic AI and Retrieval-Augmented Generation (RAG).

## Overview

This application allows you to ask questions about the Erlang BEAM virtual machine and runtime system using a RAG (Retrieval-Augmented Generation) approach. It processes the BEAM book's asciidoc chapters and code examples, creates embeddings using sentence-transformers, and stores them in ChromaDB. The Pydantic AI agent then retrieves relevant context to provide accurate, grounded answers.

## Features

- **Document Processing**: Parses asciidoc files from the BEAM book chapters and code examples
- **Vector Embeddings**: Uses sentence-transformers (all-MiniLM-L6-v2) for document embeddings
- **Vector Store**: ChromaDB with filesystem storage for efficient retrieval
- **RAG Agent**: Pydantic AI agent that retrieves context before generating answers
- **Structured Output**: Returns answers with source snippets and confidence levels
- **CLI Interface**: Easy-to-use command-line interface for indexing and querying

## Installation

### Prerequisites

- Python 3.11 or higher
- uv (Python package manager)

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
# Navigate to the rag directory
cd rag

# Install dependencies using uv
uv pip install -e ".[dev]"
```

Or install directly:

```bash
proxychains4 uv pip install pydantic-ai chromadb sentence-transformers click rich python-dotenv
```

## Configuration

### Use Anthropic Claude Models

You can use other models supported by Pydantic AI:

```bash
# Use Anthropic Claude
proxychains4 uv run beam-rag ask "What is BEAM?" --model="anthropic:claude-sonnet-4-5"
```

## Usage

### 1. Index the Documents

First, you need to index all the BEAM book documents:

```bash
pip install -U huggingface_hub
set -x HF_ENDPOINT https://hf-mirror.com # for fish shell
hf download sentence-transformers/all-MiniLM-L6-v2 ## needed for user in China, otherwise it keeps timeout to download the embeding model.

uv run beam-rag index
```

This will:
- Process all `.asciidoc` files from the `../chapters/` directory
- Process all code files from the `../code/` directory
- Create embeddings using sentence-transformers
- Store them in ChromaDB (default: `./vector_db/`)

You can specify custom paths:

```bash
uv run beam-rag index --book-path /path/to/theBeamBook --vector-db /path/to/vector_db
```

### 2. Ask Questions

Once indexed, you can ask questions about the BEAM:

```bash
uv run beam-rag ask "What is the BEAM virtual machine?"
uv run beam-rag ask "How does Erlang process scheduling work?"
uv run beam-rag ask "Explain the Erlang garbage collector"
uv run beam-rag ask "What are the main components of ERTS?"
```

The agent will:
- Retrieve relevant context from the vector store
- Generate a comprehensive answer
- Show confidence level and source snippets

### 3. Check Statistics

View statistics about the indexed documents:

```bash
uv run beam-rag stats
```

### 4. Clear the Index

To remove all indexed documents:

```bash
uv run beam-rag clear
```

## Architecture

### Components

1. **DocumentProcessor** (`document_processor.py`)
   - Parses asciidoc files and extracts sections
   - Processes code files by language
   - Chunks documents for optimal retrieval

2. **VectorStore** (`vector_store.py`)
   - Manages ChromaDB collection
   - Creates embeddings using sentence-transformers
   - Handles document search and retrieval

3. **RAGAgent** (`rag_agent.py`)
   - Pydantic AI agent with specialized system prompt
   - Retrieves context before answering
   - Returns structured output with sources

4. **CLI** (`cli.py`)
   - Command-line interface with rich output
   - Commands: index, ask, stats, clear

### Data Flow

```
User Question
    ↓
[Retrieve Context] → VectorStore (ChromaDB + Sentence Transformers)
    ↓
[Generate Answer] → Pydantic AI Agent (GPT-4o-mini)
    ↓
Structured Response (answer + sources + confidence)
```

## Project Structure

```
rag/
├── pyproject.toml              # Project configuration
├── README.md                   # This file
├── vector_db/                  # ChromaDB storage (created on first run)
├── src/
│   └── beam_rag/
│       ├── __init__.py         # Package initialization
│       ├── document_processor.py   # Document parsing and chunking
│       ├── vector_store.py         # ChromaDB and embeddings
│       ├── rag_agent.py            # Pydantic AI agent
│       └── cli.py                  # Command-line interface
└── tests/                      # Unit tests
    ├── __init__.py
    ├── conftest.py
    ├── test_document_processor.py
    ├── test_vector_store.py
    ├── test_rag_agent.py
    └── test_cli.py
```

## Example Output

```
╭──────────────────────────────────────────────────────────────╮
│ Question: What is the BEAM virtual machine?                    │
╰──────────────────────────────────────────────────────────────╯

Retrieving relevant context...

Answer:
╭──────────────────────────────────────────────────────────────╮
│ BEAM (Bogdan/Björn's Erlang Abstract Machine) is the virtual │
│ machine used for executing Erlang code. It runs BEAM         │
│ bytecode instructions within the context of an Erlang node.  │
│                                                              │
│ Key characteristics:                                         │
│ - Written in C for portability                               │
│ - Supports lightweight processes with message passing        │
│ - Includes automatic memory management                       │
│ - Has preemption and scheduling capabilities                 │
│                                                              │
│ The BEAM is part of the larger ERTS (Erlang Runtime System)  │
│ which includes additional components like I/O, networking,  │
│ and time handling.                                           │
╰──────────────────────────────────────────────────────────────╯

Confidence: high

Sources:

1. [chapters/beam.asciidoc - The Erlang Virtual Machine: BEAM]
   The BEAM virtual machine used for executing Erlang code, just like the JVM is used for executing Java code...

2. [chapters/introduction.asciidoc - ERTS]
   BEAM is the virtual machine used for executing Erlang code...
```

## Development

### Running Tests

The project includes comprehensive unit tests covering all modules:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_document_processor.py
uv run pytest tests/test_vector_store.py
uv run pytest tests/test_rag_agent.py
uv run pytest tests/test_cli.py

# Run with coverage report
uv run pytest --cov=beam_rag --cov-report=html
```

#### Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── test_document_processor.py  # Tests for DocumentChunk and DocumentProcessor
├── test_vector_store.py        # Tests for VectorStore and embeddings
├── test_rag_agent.py           # Tests for RAGResponse and RAGAgent (async)
└── test_cli.py                 # Tests for CLI commands
```

#### Test Coverage (45 tests total)

**Document Processor Tests (13 tests)**
- DocumentChunk dataclass creation and defaults
- DocumentProcessor initialization (default and custom paths)
- AsciiDoc content cleaning (bold, italic, includes, etc.)
- Text chunking (short and long text)
- Code chunking (Erlang, C, Python)
- Processing asciidoc and code files
- File discovery

**Vector Store Tests (12 tests)**
- Cached model path retrieval
- VectorStore initialization (default and with cached model)
- Document ID generation
- Document addition (empty list, duplicates)
- Search functionality (with and without results)
- Collection statistics
- Collection clearing

**RAG Agent Tests (10 tests)**
- RAGResponse model creation and validation
- RAGAgent initialization (default, custom params, Anthropic)
- Async querying (no context, with context, multiple results)
- Context building (with and without sections)
- Statistics retrieval

**CLI Tests (11 tests)**
- Help and version commands
- Index command
- Stats command
- Clear command (confirm and cancel)
- Ask command (with API key, no API key, no documents)
- Command options

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/
```

## Tips

- **Indexing Time**: Initial indexing may take a few minutes depending on your hardware
- **Storage Size**: The vector database typically requires ~100-200MB of storage
- **Query Performance**: First query may be slower as models are loaded into memory
- **Re-indexing**: Run `beam-rag index` again after pulling new book updates

## Troubleshooting

### "No documents in vector store"

Run the indexing command first:
```bash
beam-rag index
```

### "API key not found"

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Import errors

Ensure all dependencies are installed:
```bash
uv pip install -e ".[dev]"
```

## License

This project is part of the BEAM book repository. See the main repository for license information.

## References

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [The BEAM Book](https://github.com/happi/theBeamBook)

## Thanks opencode coding agent and the free kimi-k2.5 model!

I built the BEAM RAG app via `opencode -m opencode/kimi-k2.5-free`, also fixed some bug by this coding agent.
- The .venv is generated by manually run `uv venv`
- The uv.lock is generated by manually run `uv lock`
- The RAG app keeps failed to download the `sentence-transformers/all-MiniLM-L6-v2` embeding model so I set a hf mirror and downloaded the model by `hf download sentence-transformers/all-MiniLM-L6-v2` then update the code to local model path.
