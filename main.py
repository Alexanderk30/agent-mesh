#!/usr/bin/env python3
"""Agent-mesh — multi-agent orchestration runtime.

Run:
  ANTHROPIC_API_KEY=sk-... \\
  GOOGLE_API_KEY=... \\
  GOOGLE_CSE_ID=... \\
  python main.py
"""

from __future__ import annotations

import json
import os
import sys

assert os.environ.get("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY environment variable"
assert os.environ.get("GOOGLE_API_KEY"), "Set GOOGLE_API_KEY environment variable"
assert os.environ.get("GOOGLE_CSE_ID"), "Set GOOGLE_CSE_ID environment variable"

from core.message_bus import MessageBus
from core.memory_store import MemoryStore
from agents.supervisor import Supervisor
from agents.researcher import Researcher
from agents.coder import Coder
from agents.critic import Critic
from agents.reviewer import Reviewer


def main() -> None:
    # Shared infrastructure
    bus = MessageBus()
    memory = MemoryStore()

    # Instantiate agents
    supervisor = Supervisor(bus, memory)
    researcher = Researcher(bus, memory)
    coder = Coder(bus, memory)
    critic = Critic(bus)
    reviewer = Reviewer(bus)

    # Register specialists with the supervisor for synchronous dispatch
    supervisor.register_agent("researcher", researcher)
    supervisor.register_agent("coder", coder)
    supervisor.register_agent("critic", critic)
    supervisor.register_agent("reviewer", reviewer)

    # Demo task
    DEMO_TASK = (
        "Research the SEC EDGAR API and write a Python function called "
        "get_company_filings(cik: str, form_type: str) -> list[dict] that "
        "returns the 10 most recent filings of a given type for a company "
        "by CIK number. Include error handling and a usage example."
    )

    print(f"\nDemo task:\n{DEMO_TASK}\n")

    result = supervisor.run(DEMO_TASK)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
