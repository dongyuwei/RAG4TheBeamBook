"""RAG Agent implementation using Pydantic AI."""

import os
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

from beam_rag.vector_store import VectorStore


class RAGResponse(BaseModel):
    """Structured response from the RAG agent."""

    answer: str = Field(
        description="The comprehensive answer to the user's question based on the BEAM book content."
    )
    source_snippets: List[str] = Field(
        description="List of relevant source snippets used to generate the answer, including file paths and key content."
    )
    confidence: str = Field(
        description="Confidence level in the answer: 'high', 'medium', or 'low' based on relevance of retrieved context."
    )


class RAGAgent:
    """RAG Agent for answering questions about the BEAM book."""

    SYSTEM_PROMPT = """You are an expert on the Erlang BEAM virtual machine and runtime system.
Your goal is to provide accurate, detailed answers based on the BEAM book content.

Instructions:
1. Use the retrieved context to answer the question comprehensively.
2. If the context doesn't contain enough information, say so clearly.
3. Always cite your sources by including file paths and section names.
4. For code-related questions, include relevant code examples from the context.
5. Explain technical concepts clearly, assuming the user has some programming knowledge.
6. Be thorough but concise in your explanations.

When providing your answer:
- Start with a direct answer to the question
- Provide additional context and details
- Include relevant code examples if applicable
- List the sources you used"""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        model: KnownModelName = "openai:gpt-4o-mini",
        api_key: str | None = None,
    ):
        """Initialize the RAG Agent.

        Args:
            vector_store: VectorStore instance for document retrieval.
            model: The LLM model to use.
            api_key: API key for the LLM service (or from environment).
        """
        self.vector_store = vector_store or VectorStore()

        # Set API key if provided
        if api_key:
            if "openai" in model:
                os.environ["OPENAI_API_KEY"] = api_key
            elif "anthropic" in model:
                os.environ["ANTHROPIC_API_KEY"] = api_key

        # Initialize Pydantic AI Agent with structured output
        self.agent = Agent(
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            result_type=RAGResponse,
            retries=2,
        )

    async def ask(self, question: str, n_results: int = 5) -> RAGResponse:
        """Ask a question about the BEAM book.

        Args:
            question: The user's question.
            n_results: Number of context documents to retrieve.

        Returns:
            RAGResponse with answer and sources.
        """
        # Retrieve relevant context
        context_results = self.vector_store.search(question, n_results=n_results)

        if not context_results:
            return RAGResponse(
                answer="I couldn't find any relevant information in the BEAM book to answer your question. Please try rephrasing or ask about a different topic.",
                source_snippets=[],
                confidence="low",
            )

        # Build context for the agent
        context_str = self._build_context(context_results)

        # Prepare the prompt with context
        prompt = f"""Question: {question}

Retrieved Context:
{context_str}

Please provide a comprehensive answer based on the retrieved context above.
"""

        # Run the agent
        result = await self.agent.run(prompt)

        # Extract source snippets
        source_snippets = []
        for result_item in context_results:
            source_info = f"[{result_item['source']}"
            if result_item.get("section"):
                source_info += f" - {result_item['section']}"
            source_info += f"]\n{result_item['content'][:500]}..."
            source_snippets.append(source_info)

        return RAGResponse(
            answer=result.data.answer,
            source_snippets=source_snippets,
            confidence=result.data.confidence,
        )

    def _build_context(self, results: List[dict]) -> str:
        """Build a context string from search results.

        Args:
            results: List of search result dictionaries.

        Returns:
            Formatted context string.
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            part = f"\n--- Document {i} ---\n"
            part += f"Source: {result['source']}\n"
            if result.get("section"):
                part += f"Section: {result['section']}\n"
            part += f"Type: {result['chunk_type']}\n"
            part += f"Content:\n{result['content']}\n"
            context_parts.append(part)

        return "\n".join(context_parts)

    def get_stats(self) -> dict:
        """Get statistics about the vector store.

        Returns:
            Dictionary with collection statistics.
        """
        return self.vector_store.get_collection_stats()
