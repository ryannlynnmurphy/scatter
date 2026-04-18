# Scatter

A local-first AI router. Every message gets classified and sent down one of three paths — shell (regex, no tokens), local Qwen via Ollama, or Claude via the Anthropic API — and every call shows its route and estimated watt cost, so you can see where your intelligence-per-watt actually goes.

Part of ongoing research on decentralized AI infrastructure. This is v0.

## What it does

```
message in
  → launch / open / start X?   →  local:launch   (spawn app, 0 tokens)
  → disk / memory / uptime?    →  local:shell    (bash command, 0 tokens)
  → write / fix / debug ...?   →  cloud:sonnet   (Claude API, ~5W local)
  → else + prefer_local?       →  local:qwen     (Ollama, ~30W)
  → else?                      →  cloud:sonnet
```

Every call appends to `~/.scatter/ipw-log.jsonl` with route, tokens, duration, and estimated watts. A web UI at `http://127.0.0.1:8787/` shows rolling 24h totals and route badges per call.

## Install

```
git clone https://github.com/ryannlynnmurphy/scatter.git
cd scatter
python3 -m venv .venv && source .venv/bin/activate
pip install anthropic fastapi uvicorn python-dotenv
cp .env.example .env        # then paste your Anthropic API key
```

Install Ollama separately: <https://ollama.com>

```
ollama pull qwen2.5-coder:1.5b
uvicorn server:app --host 127.0.0.1 --port 8787 --reload
```

Open <http://127.0.0.1:8787>.

## Status

- **v1.1 shipped**: three routes, IPW logging, web UI, live route badges
- **Deferred**: julienne (chunked map-reduce for long local context), mobile client, real wattmeter integration, distributed cluster inference across Pi nodes

## Why

Most AI chat tools send every message to a data center. Scatter decides per-message: is this a regex, a small local model, or does it genuinely need a frontier model? Every decision is visible. Your laptop isn't a dumb terminal.

Watts are currently estimated, not metered. Cloud calls only log the laptop-side cost — data-center watts are opaque to any user. That asymmetry is the honest caveat of the thermodynamic argument.

## License

MIT.
