"""
LangGraph agent with two nodes:
  1. intent_router  — LLM classifies query as 'transactions' or 'rag'
  2. executor       — calls the right tool and generates a response
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Literal, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.rag import get_vector_store
from app.routers.banking import get_insights, get_transactions

# ── LLM ─────────────────────────────────────────────────────────────────────

def _llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        temperature=0.3,
    )


# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_id: str
    session_id: str
    message: str
    history: list[dict]
    intent: str          # 'transactions' | 'rag'
    answer: str


# ── Node 1: Intent Router ─────────────────────────────────────────────────────

ROUTER_PROMPT = """You are a financial assistant router. Given the user message, respond with ONLY ONE of these two words:
- transactions  (if the query is about spending, transactions, categories, amounts, insights, comparisons, financial data)
- rag           (if the query is about financial advice, budgeting strategies, saving tips, financial education)

User message: {message}
Response (one word only):"""


def intent_router(state: AgentState) -> AgentState:
    llm = _llm()
    prompt = ROUTER_PROMPT.format(message=state["message"])
    response = llm.invoke(prompt)
    raw = response.content.strip().lower()
    intent = "transactions" if "transaction" in raw else "rag"
    return {**state, "intent": intent}


# ── Node 2: Executor ──────────────────────────────────────────────────────────

TRANSACTION_PROMPT = """You are a helpful personal finance assistant. A user asked: "{message}"

Here is their financial data:
{data}

Conversation history:
{history}

Provide a clear, friendly, and insightful answer. Include specific numbers, percentages, and actionable observations where relevant. Be concise but complete."""


RAG_PROMPT = """You are a knowledgeable personal finance advisor. A user asked: "{message}"

Relevant financial knowledge retrieved:
{context}

Conversation history:
{history}

Provide a helpful, grounded answer based on the retrieved knowledge. Be practical and specific. If the knowledge doesn't fully cover the question, answer from general financial expertise."""


def _date_range_for_period(period: str) -> tuple[str, str]:
    """Return ISO start/end strings for common periods."""
    now = datetime.now(timezone.utc)
    if "last week" in period or "previous week" in period:
        end = now - timedelta(days=now.weekday() + 1)
        start = end - timedelta(days=6)
    elif "this week" in period:
        start = now - timedelta(days=now.weekday())
        end = now
    elif "last month" in period:
        first_this = now.replace(day=1)
        end = first_this - timedelta(days=1)
        start = end.replace(day=1)
    elif "this month" in period:
        start = now.replace(day=1)
        end = now
    else:
        # default: last 7 days
        start = now - timedelta(days=7)
        end = now
    return start.isoformat(), end.isoformat()


def _extract_period(message: str) -> tuple[str, str]:
    msg = message.lower()
    for keyword in ["last week", "this week", "last month", "this month", "previous week"]:
        if keyword in msg:
            return _date_range_for_period(keyword)
    return _date_range_for_period("last 7 days")


def executor(state: AgentState) -> AgentState:
    llm = _llm()
    history_text = "\n".join(
        f"{h['role'].capitalize()}: {h['content']}" for h in state.get("history", [])
    ) or "No prior conversation."

    if state["intent"] == "transactions":
        start, end = _extract_period(state["message"])

        # Fetch transactions and insights
        txn_data = get_transactions(user_id=state["user_id"], start_date=start, end_date=end)
        insight_data = get_insights(user_id=state["user_id"], start_date=start, end_date=end)

        # Also fetch previous period for comparison if "compare" is in message
        comparison_text = ""
        if any(w in state["message"].lower() for w in ["compare", "previous", "last week vs", "vs"]):
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            delta = end_dt - start_dt
            prev_end = start_dt
            prev_start = start_dt - delta
            prev_insight = get_insights(
                user_id=state["user_id"],
                start_date=prev_start.isoformat(),
                end_date=prev_end.isoformat(),
            )
            if prev_insight["total_spent"] > 0:
                pct = ((insight_data["total_spent"] - prev_insight["total_spent"]) / prev_insight["total_spent"]) * 100
                comparison_text = f"\n\nPrevious period total: ${prev_insight['total_spent']:.2f}. Change: {pct:+.1f}%"

        data_summary = json.dumps(
            {
                "period": {"start": start, "end": end},
                "total_spent": insight_data["total_spent"],
                "transaction_count": insight_data["transaction_count"],
                "top_category": insight_data["top_category"],
                "category_breakdown": insight_data["category_breakdown"],
                "recent_transactions": txn_data["transactions"][:8],
            },
            indent=2,
        ) + comparison_text

        prompt = TRANSACTION_PROMPT.format(
            message=state["message"],
            data=data_summary,
            history=history_text,
        )

    else:  # rag
        store = get_vector_store()
        chunks = store.retrieve(state["message"])
        context = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in chunks
        )
        prompt = RAG_PROMPT.format(
            message=state["message"],
            context=context,
            history=history_text,
        )

    response = llm.invoke(prompt)
    return {**state, "answer": response.content.strip()}


# ── Route function ────────────────────────────────────────────────────────────

def route_after_router(state: AgentState) -> Literal["executor"]:
    return "executor"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("intent_router", intent_router)
    g.add_node("executor", executor)
    g.set_entry_point("intent_router")
    g.add_conditional_edges("intent_router", route_after_router, {"executor": "executor"})
    g.add_edge("executor", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
