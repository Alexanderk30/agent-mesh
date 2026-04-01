"""Researcher agent — searches the web and synthesises findings."""

from __future__ import annotations

import json
from typing import Any

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.message_bus import MessageBus
from core.memory_store import MemoryStore
from core.types import TaskAssign, Result
from tools import web_search

MODEL = "claude-sonnet-4-5-20250514"


class Researcher:
    """Specialist agent that researches topics using web search."""

    def __init__(self, bus: MessageBus, memory: MemoryStore) -> None:
        self.bus = bus
        self.memory = memory
        self.client = anthropic.Anthropic()

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def _call_llm(self, messages: list, tools: list, tool_choice: Any = "auto") -> Any:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        usage = response.usage
        print(
            f'[RESEARCHER][{MODEL}] action="research" '
            f"input_tokens={usage.input_tokens} output_tokens={usage.output_tokens}"
        )
        return response

    def process_one(self) -> None:
        """Process one pending TaskAssign from the bus."""
        for msg in self.bus.subscribe("researcher"):
            if isinstance(msg, TaskAssign):
                self._handle(msg)
                return

    def _handle(self, task: TaskAssign) -> None:
        print(f"[RESEARCHER] Working on: {task.description[:80]}...")

        tools = [web_search.TOOL_DEFINITION]
        messages = [
            {
                "role": "user",
                "content": (
                    f"Research the following topic thoroughly. Use the web_search tool "
                    f"to find relevant information, then synthesise your findings into "
                    f"a clear, detailed summary.\n\nTopic: {task.description}"
                ),
            }
        ]

        sources: list[dict] = []
        final_text = ""

        # Agentic tool-use loop
        for _ in range(10):  # safety cap on iterations
            response = self._call_llm(messages=messages, tools=tools)

            # Collect assistant content
            assistant_content = response.content

            # Check for tool use
            tool_uses = [b for b in assistant_content if b.type == "tool_use"]

            if not tool_uses:
                # No more tool calls — extract final text
                for block in assistant_content:
                    if hasattr(block, "text"):
                        final_text += block.text
                break

            # Append assistant message
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool call and build tool_result messages
            tool_results = []
            for tu in tool_uses:
                if tu.name == "web_search":
                    results = web_search.execute(**tu.input)
                    sources.extend(results)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(results),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
        else:
            # Fell through — grab whatever text we have
            for block in assistant_content:
                if hasattr(block, "text"):
                    final_text += block.text

        # Publish result
        result = Result(
            sender="researcher",
            recipient="supervisor",
            correlation_id=task.correlation_id,
            subtask_id=task.subtask_id,
            data={
                "findings": final_text,
                "sources": [s.get("url", "") for s in sources if s.get("url")],
            },
            confidence=0.85,
            success=True,
        )
        self.bus.publish(result)
        print(f"[RESEARCHER] Completed subtask {task.subtask_id}")
