# RAG for the Erlang BEAM book
A Personal Knowledge Retriever for the Erlang BEAM book

## Overview
Build a functional, single-agent RAG (Retrieval-Augmented Generation) application. Master Pydantic AI agents and tool-based context retrieval. RAG demo code can check https://ai.pydantic.dev/examples/rag/

## Requirements
Design a single-agent system that can answer questions based on a specific local dataset (mainly all the *.asciidoc files with current ./chapters/ folder, and you can also reference to source code within ./code foler). The agent must retrieve relevant context before generating a response to ensure accuracy.

The system should demonstrate the move from a basic "knowledge-less" chatbot to one that uses external data to provide grounded answers.

## In Scope
- Use Pydantic AI: Define a specialized agent with a clear system prompt.
- Implement RAG: Create a tool (function) that the agent can call to search and retrieve relevant context from a vector db.
- Structured Outputs: Use Pydantic models to ensure the agent returns both the final answer and the source_snippet used.
- Vector db: Create embeddings from text documents, stored them in a vector database, and implemented a retriever pipeline to fetch relevant context for LLM response.

## Tips
- For creating the embeddings, we recommend using Pydantic AI’s Local Sentence Transformers.
- Recommended technologies for the vector store are: `qadrant` or `ChromaDB`, and use filesystem storage.
- All RAG relacted code should been put in ./rag folder.
- Generate a README.md in the new rag app root folder to tell me how to run the app.
- Replace all `pip` commands with `uv` commands.
- init the rag app with `uv init`, put all new codes into the new rag app folder.
- Write out the work flow, and implement the RAG system step by step. Commit changes with each step and give clean commit messages. 

## Out of Scope
- Evaluation Suites: No requirement for Pydantic Evals or formal benchmarking for this assignment.
- Multi-Agent Coordination: Focus on one high-performing agent rather than a multi-agent system.
- Persistent Memory: Maintaining history across multiple sessions is not required for this mini-task.