# Items Needing Owner Input (Optional / Deferred)

1. **Optional Cloud API Keys for Multi-Lab Live Runs:** Currently `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and `OPENROUTER_API_KEY` are unset in the environment. Add whichever keys your Apart Research team wants to evaluate into a local `.env` file (never committed) and run `apart-epistemics run --backend live ...` (or run `--backend ollama` immediately using your local `llama3.1:8b` model).
2. **Optional Claude Design Upgrade:** Run `design-briefs/epistemic-prism.md` in Claude Design (`claude.ai` -> Artifacts -> Claude Design -> Import GitHub repo -> Export -> Send to local coding agent) if you want an interactive Claude Design visual pass on top of the Part C editorial dashboard.

