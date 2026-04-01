"""Reviewer agent — final quality gate for the assembled pipeline output."""

from __future__ import annotations

import json
from typing import Any

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.message_bus import MessageBus
from core.types import TaskAssign, Approve, Reject

MODEL = "claude-opus-4-5-20250129"

REVIEW_TOOL = {
    "name": "review",
    "description": "Provide a final quality review of the assembled pipeline output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "Whether the overall output meets quality standards",
            },
            "justification": {
                "type": "string",
                "description": "Explanation of the approval or rejection decision",
            },
            "quality_score": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Overall quality score (0-1)",
            },
            "suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Suggestions for improvement",
            },
        },
        "required": ["approved", "justification", "quality_score"],
    },
}


class Reviewer:
    """Final quality gate that approves or rejects the assembled output."""

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        self.client = anthropic.Anthropic()

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def _call_llm(self, messages: list, tools: list, tool_choice: dict) -> Any:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        usage = response.usage
        print(
            f'[REVIEWER][{MODEL}] action="review" '
            f"input_tokens={usage.input_tokens} output_tokens={usage.output_tokens}"
        )
        return response

    def process_one(self) -> None:
        """Process one pending TaskAssign from the bus."""
        for msg in self.bus.subscribe("reviewer"):
            if isinstance(msg, TaskAssign):
                self._handle(msg)
                return

    def _handle(self, task: TaskAssign) -> None:
        print(f"[REVIEWER] Reviewing final output for: {task.description[:80]}...")

        assembled = json.dumps(task.context.get("assembled_output", {}), default=str)[:4000]

        messages = [
            {
                "role": "user",
                "content": (
                    f"Review the assembled output of a multi-agent pipeline.\n\n"
                    f"Original task: {task.description}\n\n"
                    f"Assembled subtask results:\n{assembled}\n\n"
                    f"Decide whether to approve or reject the overall output. "
                    f"Consider correctness, completeness, coherence, and whether "
                    f"the output adequately addresses the original task."
                ),
            }
        ]

        response = self._call_llm(
            messages=messages,
            tools=[REVIEW_TOOL],
            tool_choice={"type": "tool", "name": "review"},
        )

        review: dict = {}
        for block in response.content:
            if block.type == "tool_use" and block.name == "review":
                review = block.input
                break

        if not review:
            review = {
                "approved": True,
                "justification": "Unable to parse review — defaulting to approve",
                "quality_score": 0.5,
                "suggestions": [],
            }

        suggestions = review.get("suggestions", [])
        print(
            f"[REVIEWER] quality_score={review['quality_score']:.2f} "
            f"approved={review['approved']} suggestions={len(suggestions)}"
        )

        if review["approved"]:
            msg = Approve(
                sender="reviewer",
                recipient="supervisor",
                correlation_id=task.correlation_id,
                subtask_id=task.subtask_id,
                justification=review["justification"],
            )
        else:
            msg = Reject(
                sender="reviewer",
                recipient="supervisor",
                correlation_id=task.correlation_id,
                subtask_id=task.subtask_id,
                reason=review["justification"],
            )

        self.bus.publish(msg)
