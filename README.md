# JudgeSim-AI Hackathon judge simulator and idea evaluator

Fetchathon project (uAgents / Agentverse / ASI One).

## Current status

- `orchestrator.py`: uAgent LLM coordinator. Supports:
  - Generate Idea (routes to idea generator agent)
  - Analyse ideas via multiple judges (sequential, with progress updates)
  - General chat fallback
- `instructions_to_ai`: shared coding instructions for AI contributors.
- `venv/`: Python environment with uagents, litellm, python-dotenv.

Run: `venv/bin/python orchestrator.py`

Required env: `ASI1_API_KEY`; optional `IDEA_GENERATOR_ADDRESS`, `JUDGE_ADDRESSES`/`JUDGE1_ADDRESS`...
