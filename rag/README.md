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
uv pip install pydantic-ai chromadb sentence-transformers click rich python-dotenv
```

## Configuration

### API Key

The RAG agent uses OpenAI's GPT-4o-mini by default. You need to set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or you can pass it directly when running commands:

```bash
beam-rag ask "What is BEAM?" --api-key="your-api-key"
```

### Alternative Models

You can use other models supported by Pydantic AI:

```bash
# Use Anthropic Claude
beam-rag ask "What is BEAM?" --model="anthropic:claude-3-5-sonnet-latest"

# Use a different OpenAI model
beam-rag ask "What is BEAM?" --model="openai:gpt-4o"
```

## Usage

### 1. Index the Documents

First, you need to index all the BEAM book documents:

```bash
beam-rag index
```

This will:
- Process all `.asciidoc` files from the `../chapters/` directory
- Process all code files from the `../code/` directory
- Create embeddings using sentence-transformers
- Store them in ChromaDB (default: `./vector_db/`)

You can specify custom paths:

```bash
beam-rag index --book-path /path/to/theBeamBook --vector-db /path/to/vector_db
```

### 2. Ask Questions

Once indexed, you can ask questions about the BEAM:

```bash
beam-rag ask "What is the BEAM virtual machine?"
beam-rag ask "How does Erlang process scheduling work?"
beam-rag ask "Explain the Erlang garbage collector"
beam-rag ask "What are the main components of ERTS?"
```

The agent will:
- Retrieve relevant context from the vector store
- Generate a comprehensive answer
- Show confidence level and source snippets

### 3. Check Statistics

View statistics about the indexed documents:

```bash
beam-rag stats
```

### 4. Clear the Index

To remove all indexed documents:

```bash
beam-rag clear
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
└── src/
    └── beam_rag/
        ├── __init__.py         # Package initialization
        ├── document_processor.py   # Document parsing and chunking
        ├── vector_store.py         # ChromaDB and embeddings
        ├── rag_agent.py            # Pydantic AI agent
        └── cli.py                  # Command-line interface
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

```bash
uv run pytest
```

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
