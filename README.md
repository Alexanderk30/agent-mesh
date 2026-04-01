# agent-mesh

Multi-agent orchestration runtime with typed message passing, supervisor routing, and shared memory — built on the Anthropic Python SDK.

## What it does

agent-mesh decomposes a high-level task into subtasks, routes each one to a specialist agent, quality-checks every result through a critic/reviewer loop, and assembles the final output — all driven by Claude models over a central message bus.

A single call to `supervisor.run("your task here")` triggers the full pipeline:

```
User Task
  └─> Supervisor (Opus) ── decomposes into subtasks
        ├─> Researcher (Sonnet) ── web search + synthesis
        ├─> Coder (Sonnet) ── write + execute Python
        ├─> Critic (Sonnet) ── score each result, request revisions
        └─> Reviewer (Opus) ── approve or reject final output
```

## Architecture

### Agents

| Agent | Model | Role |
|---|---|---|
| **Supervisor** | claude-opus-4-5 | Decomposes tasks via structured tool output, topologically sorts subtasks by dependency, dispatches to specialists, and handles critic-driven retry loops (up to 3 retries per subtask). |
| **Researcher** | claude-sonnet-4-5 | Runs an agentic tool-use loop with Google Custom Search to gather sources and synthesise findings into a structured summary. |
| **Coder** | claude-sonnet-4-5 | Writes Python code informed by prior research from the memory store, executes it in a sandboxed subprocess, and iterates until tests pass. |
| **Critic** | claude-sonnet-4-5 | Scores every subtask result on correctness, completeness, and clarity (0–1 each) and returns APPROVE, REVISE, or REJECT. |
| **Reviewer** | claude-opus-4-5 | Final quality gate — reviews all assembled subtask results against the original task and issues an Approve or Reject. |

### Core infrastructure

**MessageBus** (`core/message_bus.py`) — Thread-safe publish/subscribe bus. All inter-agent communication routes exclusively through the bus; agents never import each other. Every message is logged with sender, recipient, type, and correlation ID.

**MemoryStore** (`core/memory_store.py`) — Key-value store indexed by subtask ID with vector similarity search. Uses `sentence-transformers` (all-MiniLM-L6-v2) when available, otherwise falls back to a built-in TF-IDF embedder with cosine similarity over numpy vectors.

**Typed messages** (`core/types.py`) — Frozen-false dataclasses with auto-generated UUIDs and ISO timestamps: `TaskAssign`, `Result`, `Feedback`, `Approve`, `Reject`. Every message carries a `correlation_id` for end-to-end tracing.

### Tools

**web_search** (`tools/web_search.py`) — Google Custom Search JSON API wrapper. Returns structured `[{title, url, snippet}]` results. Degrades gracefully if credentials are missing.

**code_exec** (`tools/code_exec.py`) — Sandboxed Python execution via `subprocess.run` with a 10-second timeout. Returns stdout, stderr, exit code, and a success flag.

### Key design decisions

All structured outputs use `tool_choice={"type":"tool","name":"<tool_name>"}` — no assistant-message prefilling for JSON extraction. Every `anthropic.messages.create` call is wrapped in tenacity retry with exponential backoff on `RateLimitError`. Every LLM call prints a standardised log line: `[AGENT][MODEL] action="..." input_tokens=N output_tokens=N`.

## Project structure

```
agent-mesh/
├── agents/
│   ├── __init__.py
│   ├── supervisor.py      # Task decomposition + orchestration
│   ├── researcher.py      # Web research agent
│   ├── coder.py           # Code generation + execution agent
│   ├── critic.py          # Quality scoring agent
│   └── reviewer.py        # Final approval gate
├── core/
│   ├── __init__.py
│   ├── message_bus.py     # Pub/sub message bus
│   ├── memory_store.py    # Vector-indexed result store
│   └── types.py           # Typed message dataclasses
├── tools/
│   ├── __init__.py
│   ├── web_search.py      # Google Custom Search
│   └── code_exec.py       # Sandboxed Python execution
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Google API key](https://console.cloud.google.com/apis/credentials) with Custom Search API enabled
- A [Google Custom Search Engine ID](https://programmablesearchengine.google.com/)

### Install

```bash
git clone <repo-url> && cd agent-mesh
pip install -r requirements.txt
```

Optional — install `sentence-transformers` for higher-quality similarity search in the memory store (falls back to TF-IDF without it):

```bash
pip install sentence-transformers
```

### Environment variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
export GOOGLE_CSE_ID="a1b2c3..."
```

## Usage

### Run the demo task

```bash
python main.py
```

This runs the built-in SEC EDGAR demo task — the supervisor decomposes it into research and coding subtasks, each gets critic-reviewed, and the assembled output goes through final reviewer approval.

### Use programmatically

```python
from core import MessageBus, MemoryStore
from agents import Supervisor, Researcher, Coder, Critic, Reviewer

bus = MessageBus()
memory = MemoryStore()

supervisor = Supervisor(bus, memory)
supervisor.register_agent("researcher", Researcher(bus, memory))
supervisor.register_agent("coder", Coder(bus, memory))
supervisor.register_agent("critic", Critic(bus))
supervisor.register_agent("reviewer", Reviewer(bus))

result = supervisor.run("Your task description here")
# result["approved"]  -> bool
# result["result"]    -> dict of subtask outputs
```

### Reading the output

The returned dict contains:

| Key | Type | Description |
|---|---|---|
| `task` | str | The original task description |
| `result` | dict | Subtask ID -> output data for every completed subtask |
| `approved` | bool | Whether the reviewer approved the final output |
| `subtask_count` | int | Number of subtasks the supervisor created |
| `correlation_id` | str | UUID for end-to-end message tracing |
| `review_justification` | str | The reviewer's explanation for approval/rejection |

## How the pipeline works

1. **Decomposition** — The supervisor sends the task to Claude Opus with a `decompose_task` tool, which returns a list of subtasks with agent assignments and dependency edges.

2. **Topological sort** — Subtasks are ordered by their `depends_on` fields via DFS so that dependencies execute first.

3. **Dispatch loop** — For each subtask in order:
   - The supervisor publishes a `TaskAssign` to the designated agent's queue
   - The agent processes it (researcher searches the web; coder writes and runs code)
   - The agent publishes a `Result` back to the supervisor
   - The supervisor forwards the result to the critic
   - The critic scores it and returns `APPROVE`, `REVISE`, or `REJECT`
   - On `REVISE`, the supervisor retries (up to 3 times) with feedback appended to the task description

4. **Assembly** — All subtask results are gathered from the memory store into a single dict.

5. **Final review** — The assembled output is sent to the reviewer (Claude Opus), which issues a final `Approve` or `Reject` with a justification and quality score.

## Adding a new agent

1. Create `agents/your_agent.py` with a class that takes `bus` (and optionally `memory`) in `__init__`
2. Implement `process_one()` — pull one `TaskAssign` from `bus.subscribe("your_agent")`, do work, publish a `Result`
3. Register it in `main.py`: `supervisor.register_agent("your_agent", YourAgent(bus, memory))`
4. Add `"your_agent"` to the `required_agent` enum in the supervisor's `DECOMPOSE_TOOL` schema

## Adding a new tool

1. Create `tools/your_tool.py` with a `TOOL_DEFINITION` dict (Anthropic tool schema) and an `execute(**kwargs)` function
2. Import it in the agent that needs it and add `TOOL_DEFINITION` to that agent's `tools` list

## License

MIT
