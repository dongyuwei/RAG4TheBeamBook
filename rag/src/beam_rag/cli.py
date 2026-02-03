"""CLI interface for the BEAM Book RAG application."""

import asyncio
import os
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from beam_rag.document_processor import DocumentProcessor
from beam_rag.rag_agent import RAGAgent
from beam_rag.vector_store import VectorStore


console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """BEAM Book RAG - Personal Knowledge Retriever for the Erlang BEAM book.

    A RAG (Retrieval-Augmented Generation) application that answers questions
    about the Erlang BEAM virtual machine using Pydantic AI.
    """
    pass


@cli.command()
@click.option(
    "--book-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to the BEAM book repository (default: parent of rag directory)",
)
@click.option(
    "--vector-db",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to store vector database (default: ./vector_db)",
)
def index(book_path: Path | None, vector_db: Path | None) -> None:
    """Index all BEAM book documents into the vector store.

    This command processes all asciidoc files from the chapters/ directory
    and code files from the code/ directory, creating embeddings and
    storing them in ChromaDB.
    """
    console.print(
        Panel.fit(
            "[bold blue]BEAM Book RAG - Indexing Documents[/bold blue]",
            border_style="blue",
        )
    )

    # Initialize components
    processor = DocumentProcessor(book_root=book_path)
    vector_store = VectorStore(persist_directory=vector_db)

    # Process documents
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Scanning documents...", total=None)

        chunks = processor.process_all()
        progress.update(
            task, description=f"[cyan]Found {len(chunks)} document chunks[/cyan]"
        )

        task2 = progress.add_task("[green]Adding to vector store...", total=None)
        added_count = vector_store.add_documents(chunks)
        progress.update(
            task2, description=f"[green]Added {added_count} new documents[/green]"
        )

    # Show stats
    stats = vector_store.get_collection_stats()
    console.print("\n[bold green]Indexing Complete![/bold green]")

    table = Table(title="Vector Store Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Collection Name", stats["collection_name"])
    table.add_row("Total Documents", str(stats["document_count"]))
    table.add_row("Vector DB Path", stats["persist_directory"])
    table.add_row("Embedding Dimension", str(stats["embedding_model"]))

    console.print(table)


@cli.command()
@click.argument("question")
@click.option(
    "--vector-db",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to vector database (default: ./vector_db)",
)
@click.option(
    "--model",
    default="openai:gpt-4o-mini",
    help="LLM model to use (default: openai:gpt-4o-mini)",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="API key for the LLM service (or set OPENAI_API_KEY env var)",
)
@click.option(
    "--n-results",
    default=5,
    type=int,
    help="Number of context documents to retrieve (default: 5)",
)
def ask(
    question: str,
    vector_db: Path | None,
    model: str,
    api_key: str | None,
    n_results: int,
) -> None:
    """Ask a question about the BEAM book.

    Example:
        beam-rag ask "What is the BEAM virtual machine?"
        beam-rag ask "How does Erlang process scheduling work?"
    """
    # Check for API key
    if not api_key and "openai" in model.lower():
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "anthropic" in model.lower():
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key and "openai" in model.lower():
        console.print(
            "[bold red]Error:[/bold red] OpenAI API key not found. "
            "Set OPENAI_API_KEY environment variable or use --api-key option."
        )
        return

    console.print(
        Panel.fit(
            f"[bold blue]Question:[/bold blue] {question}",
            border_style="blue",
        )
    )

    # Initialize agent
    vector_store = VectorStore(persist_directory=vector_db)
    agent = RAGAgent(
        vector_store=vector_store,
        model=model,
        api_key=api_key,
    )

    # Check if vector store has documents
    stats = agent.get_stats()
    if stats["document_count"] == 0:
        console.print(
            "[bold yellow]Warning:[/bold yellow] No documents in vector store. "
            "Please run 'beam-rag index' first."
        )
        return

    # Run the query
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("[cyan]Retrieving relevant context...", total=None)

        # Run async query
        response = asyncio.run(agent.ask(question, n_results=n_results))

    # Display results
    console.print("\n[bold green]Answer:[/bold green]")
    console.print(Panel(response.answer, border_style="green"))

    # Show confidence
    confidence_color = {
        "high": "green",
        "medium": "yellow",
        "low": "red",
    }.get(response.confidence, "white")
    console.print(
        f"\n[bold]Confidence:[/bold] [{confidence_color}]{response.confidence}[/{confidence_color}]"
    )

    # Show sources
    if response.source_snippets:
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        for i, source in enumerate(response.source_snippets[:3], 1):
            console.print(f"\n[dim]{i}.[/dim] {source}")


@cli.command()
@click.option(
    "--vector-db",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to vector database (default: ./vector_db)",
)
def stats(vector_db: Path | None) -> None:
    """Show statistics about the indexed documents."""
    vector_store = VectorStore(persist_directory=vector_db)
    stats_data = vector_store.get_collection_stats()

    console.print(
        Panel.fit(
            "[bold blue]BEAM Book RAG - Statistics[/bold blue]",
            border_style="blue",
        )
    )

    table = Table(title="Vector Store Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Collection Name", stats_data["collection_name"])
    table.add_row("Total Documents", str(stats_data["document_count"]))
    table.add_row("Vector DB Path", stats_data["persist_directory"])
    table.add_row("Embedding Dimension", str(stats_data["embedding_model"]))

    console.print(table)


@cli.command()
@click.option(
    "--vector-db",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to vector database (default: ./vector_db)",
)
@click.confirmation_option(
    prompt="Are you sure you want to clear all indexed documents?"
)
def clear(vector_db: Path | None) -> None:
    """Clear all indexed documents from the vector store."""
    vector_store = VectorStore(persist_directory=vector_db)
    vector_store.clear()
    console.print("[bold green]Vector store cleared successfully.[/bold green]")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
