"""Typed message dataclasses for inter-agent communication."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Literal


def _new_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=False)
class BaseMessage:
    """Base class for all inter-agent messages."""

    sender: str
    recipient: str
    correlation_id: str
    msg_id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=False)
class TaskAssign(BaseMessage):
    """Supervisor assigns a subtask to a specialist agent."""

    subtask_id: str = ""
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=False)
class Result(BaseMessage):
    """Agent reports the result of a completed subtask."""

    subtask_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    success: bool = True


@dataclass(frozen=False)
class Feedback(BaseMessage):
    """Critic provides scored feedback on a subtask result."""

    subtask_id: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    recommendation: Literal["APPROVE", "REVISE", "REJECT"] = "APPROVE"
    feedback_text: str = ""


@dataclass(frozen=False)
class Approve(BaseMessage):
    """Reviewer approves the final assembled output."""

    subtask_id: str = ""
    justification: str = ""


@dataclass(frozen=False)
class Reject(BaseMessage):
    """Reviewer rejects the final assembled output."""

    subtask_id: str = ""
    reason: str = ""
