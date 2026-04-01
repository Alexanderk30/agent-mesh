"""Supervisor agent — decomposes tasks, orchestrates the agent pipeline."""

from __future__ import annotations

import json
from typing import Any, Dict, List
from uuid import uuid4

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.message_bus import MessageBus
from core.memory_store import MemoryStore
from core.types import TaskAssign, Result, Feedback, Approve, Reject

MODEL = "claude-opus-4-5-20250129"
MAX_RETRIES_PER_SUBTASK = 3

DECOMPOSE_TOOL = {
    "name": "decompose_task",
    "description": "Break a high-level task into ordered subtasks assigned to specialist agents.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subtask_id": {"type": "string"},
                        "description": {"type": "string"},
                        "required_agent": {
                            "type": "string",
                            "enum": ["researcher", "coder", "critic", "reviewer"],
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "subtask_id",
                        "description",
                        "required_agent",
                        "depends_on",
                    ],
                },
            }
        },
        "required": ["subtasks"],
    },
}


class Supervisor:
    """Orchestrator that decomposes tasks and drives the agent pipeline."""

    def __init__(self, bus: MessageBus, memory: MemoryStore) -> None:
        self.bus = bus
        self.memory = memory
        self.client = anthropic.Anthropic()

    # ------------------------------------------------------------------ #
    # LLM call with retry
    # ------------------------------------------------------------------ #
    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def _call_llm(self, messages: list, tools: list, tool_choice: dict) -> Any:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        usage = response.usage
        print(
            f'[SUPERVISOR][{MODEL}] action="decompose_task" '
            f"input_tokens={usage.input_tokens} output_tokens={usage.output_tokens}"
        )
        return response

    # ------------------------------------------------------------------ #
    # Task decomposition
    # ------------------------------------------------------------------ #
    def decompose_task(self, task: str, correlation_id: str) -> List[dict]:
        """Use the LLM to split *task* into a list of subtask dicts."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"Decompose this task into subtasks for specialist agents.\n\n"
                    f"Available agents:\n"
                    f"- researcher: searches the web for information\n"
                    f"- coder: writes and executes Python code\n"
                    f"- critic: evaluates quality of results (auto-assigned, do NOT include)\n"
                    f"- reviewer: final quality gate (auto-assigned, do NOT include)\n\n"
                    f"Task: {task}"
                ),
            }
        ]
        response = self._call_llm(
            messages=messages,
            tools=[DECOMPOSE_TOOL],
            tool_choice={"type": "tool", "name": "decompose_task"},
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "decompose_task":
                return block.input["subtasks"]

        raise RuntimeError("Supervisor did not produce a decompose_task tool call")

    # ------------------------------------------------------------------ #
    # Topological sort
    # ------------------------------------------------------------------ #
    @staticmethod
    def _topo_sort(subtasks: List[dict]) -> List[dict]:
        index = {s["subtask_id"]: s for s in subtasks}
        visited: set = set()
        order: list = []

        def dfs(sid: str) -> None:
            if sid in visited:
                return
            visited.add(sid)
            for dep in index[sid].get("depends_on", []):
                if dep in index:
                    dfs(dep)
            order.append(index[sid])

        for s in subtasks:
            dfs(s["subtask_id"])
        return order

    # ------------------------------------------------------------------ #
    # Wait helpers
    # ------------------------------------------------------------------ #
    def _wait_for(self, msg_type: type, subtask_id: str, timeout_polls: int = 300) -> Any:
        """Busy-wait for a specific message type addressed to 'supervisor'."""
        import time

        for _ in range(timeout_polls):
            for msg in self.bus.subscribe("supervisor"):
                if isinstance(msg, msg_type) and msg.subtask_id == subtask_id:
                    return msg
                # Re-queue messages that aren't what we're waiting for
                self.bus.publish(msg.__class__(**{**msg.to_dict(), "recipient": "supervisor"}))
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for {msg_type.__name__} on {subtask_id}")

    # ------------------------------------------------------------------ #
    # Main orchestration loop
    # ------------------------------------------------------------------ #
    def run(self, task: str) -> Dict[str, Any]:
        correlation_id = str(uuid4())
        print(f"\n{'='*60}")
        print(f"[SUPERVISOR] Starting task: {task[:80]}...")
        print(f"[SUPERVISOR] correlation_id={correlation_id[:8]}...")
        print(f"{'='*60}\n")

        # 1. Decompose
        subtasks = self.decompose_task(task, correlation_id)
        print(f"[SUPERVISOR] Decomposed into {len(subtasks)} subtask(s)")

        # 2. Topological sort
        sorted_tasks = self._topo_sort(subtasks)

        # 3. Execute each subtask
        for subtask in sorted_tasks:
            sid = subtask["subtask_id"]
            agent = subtask["required_agent"]
            desc = subtask["description"]
            retry_count = 0

            while retry_count <= MAX_RETRIES_PER_SUBTASK:
                print(f"\n[SUPERVISOR] Dispatching subtask {sid} -> {agent}")

                # a. Publish TaskAssign
                task_msg = TaskAssign(
                    sender="supervisor",
                    recipient=agent,
                    correlation_id=correlation_id,
                    subtask_id=sid,
                    description=desc,
                    context={"retry": retry_count},
                )
                self.bus.publish(task_msg)

                # b. Let the agent process it
                self._run_agent(agent)

                # c. Wait for Result
                result_msg = self._wait_for(Result, sid)
                self.memory.put(sid, result_msg.data)

                # d. Send result to critic
                critic_result = Result(
                    sender="supervisor",
                    recipient="critic",
                    correlation_id=correlation_id,
                    subtask_id=sid,
                    data=result_msg.data,
                    confidence=result_msg.confidence,
                    success=result_msg.success,
                )
                self.bus.publish(critic_result)
                self._run_agent("critic")

                # e. Wait for Feedback
                feedback = self._wait_for(Feedback, sid)
                print(
                    f"[SUPERVISOR] Critic feedback for {sid}: "
                    f"{feedback.recommendation} (scores={feedback.scores})"
                )

                if feedback.recommendation == "APPROVE":
                    break
                elif feedback.recommendation == "REVISE" and retry_count < MAX_RETRIES_PER_SUBTASK:
                    retry_count += 1
                    desc = f"{subtask['description']}\n\nPrevious feedback: {feedback.feedback_text}"
                    print(f"[SUPERVISOR] Retrying subtask {sid} (attempt {retry_count})")
                else:
                    print(f"[SUPERVISOR] WARNING: subtask {sid} — {feedback.recommendation}, moving on")
                    break

        # 4. Assemble final output
        final_output: Dict[str, Any] = {}
        for subtask in sorted_tasks:
            sid = subtask["subtask_id"]
            stored = self.memory.get(sid)
            if stored:
                final_output[sid] = stored

        # 5. Send to reviewer
        review_task = TaskAssign(
            sender="supervisor",
            recipient="reviewer",
            correlation_id=correlation_id,
            subtask_id="final_review",
            description=task,
            context={"assembled_output": final_output},
        )
        self.bus.publish(review_task)
        self._run_agent("reviewer")

        # 6. Wait for Approve or Reject
        approved = False
        justification = ""
        import time

        for _ in range(300):
            for msg in self.bus.subscribe("supervisor"):
                if isinstance(msg, Approve) and msg.subtask_id == "final_review":
                    approved = True
                    justification = msg.justification
                    break
                elif isinstance(msg, Reject) and msg.subtask_id == "final_review":
                    approved = False
                    justification = msg.reason
                    break
            if justification:
                break
            time.sleep(0.05)

        result = {
            "task": task,
            "result": final_output,
            "approved": approved,
            "subtask_count": len(subtasks),
            "correlation_id": correlation_id,
            "review_justification": justification,
        }

        status = "APPROVED" if approved else "REJECTED"
        print(f"\n{'='*60}")
        print(f"[SUPERVISOR] Pipeline complete — {status}")
        print(f"{'='*60}\n")

        return result

    # ------------------------------------------------------------------ #
    # Agent dispatch (synchronous — agents run in the same thread)
    # ------------------------------------------------------------------ #
    def _run_agent(self, agent_name: str) -> None:
        """Invoke the named agent's processing loop once."""
        if not hasattr(self, "_agents"):
            self._agents: Dict[str, Any] = {}
        agent = self._agents.get(agent_name)
        if agent is None:
            raise RuntimeError(f"Agent '{agent_name}' not registered with Supervisor")
        agent.process_one()

    def register_agent(self, name: str, agent: Any) -> None:
        """Register a specialist agent for synchronous dispatch."""
        if not hasattr(self, "_agents"):
            self._agents: Dict[str, Any] = {}
        self._agents[name] = agent
