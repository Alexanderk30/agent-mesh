"""Coder agent — writes and executes Python code."""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.message_bus import MessageBus
from core.memory_store import MemoryStore
from core.types import TaskAssign, Result
from tools import code_exec

MODEL = "claude-sonnet-4-5-20250514"


class Coder:
    """Specialist agent that writes and tests Python code."""

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
            f'[CODER][{MODEL}] action="code" '
            f"input_tokens={usage.input_tokens} output_tokens={usage.output_tokens}"
        )
        return response

    def process_one(self) -> None:
        """Process one pending TaskAssign from the bus."""
        for msg in self.bus.subscribe("coder"):
            if isinstance(msg, TaskAssign):
                self._handle(msg)
                return

    @staticmethod
    def _extract_python_blocks(text: str) -> str:
        """Pull ```python ... ``` fenced code blocks from assistant text."""
        blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        return "\n\n".join(blocks) if blocks else text

    def _handle(self, task: TaskAssign) -> None:
        print(f"[CODER] Working on: {task.description[:80]}...")

        # Gather prior research for context
        prior = self.memory.find_similar(task.description, top_k=2)
        context = json.dumps(prior, default=str) if prior else "No prior research available"

        tools = [code_exec.TOOL_DEFINITION]
        messages = [
            {
                "role": "user",
                "content": (
                    f"Write Python code to accomplish the following task. "
                    f"Use the code_exec tool to test your code and iterate until "
                    f"it runs correctly.\n\n"
                    f"Task: {task.description}\n\n"
                    f"Prior research context:\n{context}"
                ),
            }
        ]

        last_exec_result: dict = {}
        final_text = ""
        assistant_content: Any = []

        # Agentic tool-use loop
        for _ in range(10):
            response = self._call_llm(messages=messages, tools=tools)
            assistant_content = response.content

            tool_uses = [b for b in assistant_content if b.type == "tool_use"]

            if not tool_uses:
                for block in assistant_content:
                    if hasattr(block, "text"):
                        final_text += block.text
                break

            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for tu in tool_uses:
                if tu.name == "code_exec":
                    exec_result = code_exec.execute(**tu.input)
                    last_exec_result = exec_result
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(exec_result),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
        else:
            for block in assistant_content:
                if hasattr(block, "text"):
                    final_text += block.text

        # Extract final code
        extracted_code = self._extract_python_blocks(final_text)

        result = Result(
            sender="coder",
            recipient="supervisor",
            correlation_id=task.correlation_id,
            subtask_id=task.subtask_id,
            data={
                "code": extracted_code,
                "language": "python",
                "test_output": last_exec_result,
                "explanation": final_text[:500],
            },
            confidence=0.8,
            success=last_exec_result.get("success", True),
        )
        self.bus.publish(result)
        print(f"[CODER] Completed subtask {task.subtask_id}")
