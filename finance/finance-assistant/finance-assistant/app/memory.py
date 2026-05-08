"""Simple in-memory conversation store keyed by session_id."""

from __future__ import annotations

from collections import defaultdict
from typing import List

_sessions: dict[str, List[dict]] = defaultdict(list)

MAX_HISTORY = 10  # keep last N turns to avoid token blowout


def get_history(session_id: str) -> List[dict]:
    return _sessions[session_id][-MAX_HISTORY:]


def add_turn(session_id: str, role: str, content: str):
    _sessions[session_id].append({"role": role, "content": content})
