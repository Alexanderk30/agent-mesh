"""Thread-safe message bus for inter-agent communication."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, Iterator, List

from .types import BaseMessage


class MessageBus:
    """Central publish/subscribe message bus.

    All inter-agent communication routes through this bus.
    Agents publish messages addressed to a recipient and subscribe
    to their own queue to receive messages.
    """

    def __init__(self) -> None:
        self._queues: Dict[str, List[BaseMessage]] = defaultdict(list)
        self._lock = threading.Lock()

    def publish(self, message: BaseMessage) -> None:
        """Append *message* to the recipient's queue."""
        with self._lock:
            self._queues[message.recipient].append(message)
        print(
            f"[BUS] {message.sender} -> {message.recipient}  "
            f"type={type(message).__name__}  "
            f"correlation={message.correlation_id[:8]}..."
        )

    def subscribe(self, agent_name: str) -> Iterator[BaseMessage]:
        """Yield and remove pending messages for *agent_name*."""
        while True:
            with self._lock:
                queue = self._queues.get(agent_name, [])
                if queue:
                    msg = queue.pop(0)
                else:
                    msg = None
            if msg is None:
                return
            yield msg

    def get_history(self, correlation_id: str) -> List[BaseMessage]:
        """Return every message matching *correlation_id* across all queues."""
        results: List[BaseMessage] = []
        with self._lock:
            for queue in self._queues.values():
                for msg in queue:
                    if msg.correlation_id == correlation_id:
                        results.append(msg)
        return results

    def clear(self, agent_name: str) -> None:
        """Drop all pending messages for *agent_name*."""
        with self._lock:
            self._queues[agent_name] = []
