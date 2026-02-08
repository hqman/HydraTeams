# HydraTeams

> **The model doesn't matter. The orchestration does.**

A translation proxy that lets Claude Code Agent Teams use any AI model as a teammate. GPT Codex, Gemini, Ollama: they all become full Claude Code agents with 15+ tools, file access, bash, git, and autonomous task execution.

One proxy. One env var. Any model.

**Status:** Working. OpenAI + ChatGPT subscription providers tested end-to-end.

```
$ hydra-proxy --model gpt-5.3-codex --provider chatgpt --passthrough lead

╔══════════════════════════════════════════╗
║           HydraProxy v0.1.0              ║
╠══════════════════════════════════════════╣
║  Port:        3456                       ║
║  Target:      gpt-5.3-codex             ║
║  Spoofing as: claude-sonnet-4-5-20250929 ║
║  Passthrough: lead                       ║
╚══════════════════════════════════════════╝

Ready. Set ANTHROPIC_BASE_URL=http://localhost:3456 on teammate processes.
```

## How It Works

Claude Code Agent Teams spawns teammates as separate Claude Code processes. Each teammate communicates with its LLM via the Anthropic Messages API. HydraTeams is a proxy that intercepts these API calls and translates them to any provider's format.

The teammate is still a **full Claude Code instance** with every tool — Read, Write, Edit, Bash, Glob, Grep, Git. It just doesn't know its brain is GPT instead of Claude.

```
┌─────────────────────┐
│   Lead Agent        │    Real Claude (passthrough)
│   (Claude Opus)     │    Detected via hydra:lead marker
│   Spawns teammates  │
└──────────┬──────────┘
           │
           │  ANTHROPIC_BASE_URL=http://localhost:3456
           │
┌──────────▼──────────┐
│  Teammate Process   │    Full Claude Code instance
│  (Claude Code CLI)  │    15+ tools, file access, bash
│  All tools work     │    Thinks it's calling Anthropic
└──────────┬──────────┘
           │
           │  POST /v1/messages (Anthropic format)
           │
┌──────────▼──────────┐
│    HydraProxy       │    Translates API formats
│    localhost:3456    │    Anthropic ↔ OpenAI / ChatGPT
│                     │    Streams SSE both ways
└──────────┬──────────┘
           │
           │  Chat Completions or Responses API
           │
┌──────────▼──────────┐
│  GPT-5.3 Codex      │    Any model, any provider
│  (or GPT-4o, etc.)  │    Zero cost via subscription
└─────────────────────┘
```

## Why HydraTeams?

**You already have the best agent framework.** Claude Code Agent Teams is a battle-tested multi-agent system with agentic loops, 15+ tools, file-based coordination, task dependency graphs, messaging, plan approval, and graceful shutdown. Building another one is reinventing the wheel. HydraTeams makes Agent Teams model-agnostic instead.

**Real cost savings.** Not every task needs a $15/M token frontier model. Route research to Gemini Flash ($0.01), codegen to Codex ($0.12), architecture to Opus ($0.15). Same team, smart routing, real savings. Or use your ChatGPT Plus subscription and pay **$0 extra**.

**Zero vendor lock-in.** If OpenAI is down, route through Gemini. If prices change, switch. New model drops? Update one config value.

**Every tool, every capability.** Unlike lightweight agent wrappers, each teammate is a full Claude Code instance. It reads code, writes code, runs tests, uses git, searches files — everything Claude Code does. The proxy only replaces the LLM brain, not the body.

## Quick Start

```bash
# Clone and build
git clone https://github.com/Pickle-Pixel/HydraTeams.git
cd HydraTeams
npm install
npm run build
```

### Option A: ChatGPT Plus Subscription (zero cost)

```bash
# One-time auth (if you haven't already)
codex --login

# Start the proxy
node dist/index.js --model gpt-5.3-codex --provider chatgpt --port 3456 --passthrough lead
```

### Option B: OpenAI API Key

```bash
export OPENAI_API_KEY=sk-...
node dist/index.js --model gpt-4o-mini --provider openai --port 3456 --passthrough lead
```

### Using with Claude Code

Add `<!-- hydra:lead -->` to your project's `CLAUDE.md` (this tells the proxy which requests are from the lead agent and should passthrough to real Claude).

```bash
# Set the env var and use Claude Code normally
export ANTHROPIC_BASE_URL=http://localhost:3456
claude
```

The lead runs on real Claude (passthrough). Spawned teammates run on GPT (translated). All tools work.

## Mixed Team Routing

HydraTeams supports running the lead on your Claude subscription while teammates use a different model:

- **Lead agent** → Detected by `<!-- hydra:lead -->` marker in CLAUDE.md system prompt → Passthrough to real Anthropic API with your subscription auth headers
- **Teammates** → No marker detected → Translated to target model (GPT, Gemini, etc.)

No API keys needed for the lead — the proxy relays your Claude subscription auth headers directly.

## CLI Options

| Flag | Env Var | Default | Description |
|---|---|---|---|
| `--model` | `HYDRA_TARGET_MODEL` | (required) | Target model for teammates |
| `--provider` | `HYDRA_TARGET_PROVIDER` | `openai` | Provider: `openai`, `chatgpt` |
| `--port` | `HYDRA_PROXY_PORT` | `3456` | Proxy listen port |
| `--spoof` | `HYDRA_SPOOF_MODEL` | `claude-sonnet-4-5-20250929` | Model name reported to Claude Code |
| `--passthrough` | `HYDRA_PASSTHROUGH` | (none) | Passthrough mode: `lead`, `*`, or comma-separated model names |

## Supported Providers

### ChatGPT Backend (`--provider chatgpt`)

Uses your ChatGPT Plus subscription via the backend Responses API. Auto-reads auth from `~/.codex/auth.json` (run `codex --login` first).

Available models: `gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.2-codex`, `gpt-5.3-codex`, `gpt-5-codex-mini`, `gpt-5.1-codex-mini`

### OpenAI API (`--provider openai`)

Standard OpenAI Chat Completions API. Requires `OPENAI_API_KEY` env var or codex auth fallback.

Available models: `gpt-4o`, `gpt-4o-mini`, `o3-mini`, etc.

## What Works Today

- **OpenAI Chat Completions** — GPT-4o, GPT-4o-mini via API key
- **ChatGPT Subscription** — GPT-5.3-codex, GPT-5-codex, etc. via ChatGPT Plus ($0 extra cost)
- **Mixed team routing** — Lead on real Claude (passthrough), teammates on GPT (translated)
- **System prompt marker** — `<!-- hydra:lead -->` in CLAUDE.md for clean lead/teammate detection
- **Subscription auth relay** — No API keys needed for lead passthrough
- **Full agentic tool loops** — Read, Write, Glob, Bash all verified working through proxy
- **Retry with backoff** — Handles 429 rate limits gracefully (5 retries, exponential backoff)
- **Non-streaming support** — Handles Claude Code's haiku warmup requests
- **Token count estimation** — Handles `/v1/messages/count_tokens` endpoint

## Project Structure

```
src/
├── index.ts                    Entry point, ASCII banner
├── proxy.ts                    HTTP server, 3-way routing, passthrough
├── config.ts                   CLI args, env vars, codex JWT auth
└── translators/
    ├── types.ts                TypeScript interfaces (Anthropic + OpenAI)
    ├── request.ts              Anthropic → OpenAI Chat Completions
    ├── messages.ts             Message history translation
    ├── response.ts             OpenAI Chat Completions SSE → Anthropic SSE
    ├── request-responses.ts    Anthropic → ChatGPT Responses API
    └── response-responses.ts   Responses API SSE → Anthropic SSE
```

Zero runtime dependencies. TypeScript + Node.js builtins only.

## Roadmap

- Google Gemini translator
- Ollama translator (mostly OpenAI-compatible)
- npm publish for `npx hydra-proxy` one-liner
- Token usage tracking and cost reporting
- Multi-proxy mode (different models per teammate)

## Documentation

| Document | Description |
|----------|-------------|
| [JOURNEY.md](JOURNEY.md) | The full build story — architecture pivots, debugging, subscription hack |
| [VISION.md](VISION.md) | Why translation beats custom frameworks |
| [PRINCIPLES.md](PRINCIPLES.md) | Core beliefs guiding every decision |
| [Architecture](architecture/ARCHITECTURE.md) | Technical spec — API translation maps, SSE stream handling |

## Built With

- [Claude Code Agent Teams](https://docs.anthropic.com/en/docs/claude-code) — The agent framework we make model-agnostic
- [HydraMCP](https://github.com/Pickle-Pixel/HydraMCP) — Multi-model AI orchestration (sister project)
- TypeScript / Node.js — zero external runtime dependencies

## License

[MIT](LICENSE)
