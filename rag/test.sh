# index the BEAM book first if not
# uv run beam-rag index

proxychains4 uv run beam-rag ask "What is BEAM?" --model="anthropic:claude-sonnet-4-5"
