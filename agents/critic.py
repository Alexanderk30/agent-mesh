"""Critic agent — evaluates subtask results on quality dimensions."""

from __future__ import annotations

import json
from typing import Any

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.message_bus import MessageBus
from core.types import Result, Feedback

MODEL = "claude-sonnet-4-5-20250514"

EVALUATE_TOOL = {
    "name": "evaluate",
    "description": "Score a subtask result on correctness, completeness, and clarity.",
    "input_schema": {
        "type": "object",
        "properties": {
            "correctness": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "How factually correct is the result (0-1)",
            },
            "completeness": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "How thoroughly does the result address the task (0-1)",
            },
            "clarity": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "How clearly is the result communicated (0-1)",
            },
            "feedback": {
                "type": "string",
                "description": "Detailed feedback explaining the scores and any suggestions",
            },
            "recommendation": {
                "type": "string",
                "enum": ["APPROVE", "REVISE", "REJECT"],
                "description": "Overall recommendation for this result",
            },
        },
        "required": ["correctness", "completeness", "clarity", "feedback", "recommendation"],
    },
}


class Critic:
    """Quality-assurance agent that scores and provides feedback on results."""

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
            f'[CRITIC][{MODEL}] action="evaluate" '
            f"input_tokens={usage.input_tokens} output_tokens={usage.output_tokens}"
        )
        return response

    def process_one(self) -> None:
        """Process one pending Result from the bus."""
        for msg in self.bus.subscribe("critic"):
            if isinstance(msg, Result):
                self._handle(msg)
                return

    def _handle(self, result: Result) -> None:
        print(f"[CRITIC] Evaluating subtask {result.subtask_id}")

        data_summary = json.dumps(result.data, default=str)[:2000]

        messages = [
            {
                "role": "user",
                "content": (
                    f"Evaluate the following subtask result for quality.\n\n"
                    f"Subtask ID: {result.subtask_id}\n"
                    f"Confidence reported by agent: {result.confidence}\n"
                    f"Success: {result.success}\n\n"
                    f"Result data:\n{data_summary}\n\n"
                    f"Score each dimension 0-1 and provide a recommendation."
                ),
            }
        ]

        response = self._call_llm(
            messages=messages,
            tools=[EVALUATE_TOOL],
            tool_choice={"type": "tool", "name": "evaluate"},
        )

        eval_result: dict = {}
        for block in response.content:
            if block.type == "tool_use" and block.name == "evaluate":
                eval_result = block.input
                break

        if not eval_result:
            eval_result = {
                "correctness": 0.5,
                "completeness": 0.5,
                "clarity": 0.5,
                "feedback": "Unable to parse evaluation",
                "recommendation": "APPROVE",
            }

        overall = (
            eval_result["correctness"]
            + eval_result["completeness"]
            + eval_result["clarity"]
        ) / 3.0

        print(
            f"[CRITIC] Scores: correctness={eval_result['correctness']:.2f} "
            f"completeness={eval_result['completeness']:.2f} "
            f"clarity={eval_result['clarity']:.2f} "
            f"overall={overall:.2f} -> {eval_result['recommendation']}"
        )

        feedback = Feedback(
            sender="critic",
            recipient="supervisor",
            correlation_id=result.correlation_id,
            subtask_id=result.subtask_id,
            scores={
                "correctness": eval_result["correctness"],
                "completeness": eval_result["completeness"],
                "clarity": eval_result["clarity"],
                "overall": overall,
            },
            recommendation=eval_result["recommendation"],
            feedback_text=eval_result["feedback"],
        )
        self.bus.publish(feedback)
