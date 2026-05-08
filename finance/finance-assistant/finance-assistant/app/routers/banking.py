import json
import os
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(tags=["Banking"])

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "transactions.json")


def load_transactions():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def parse_dt(dt_str: str) -> datetime:
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@router.get("/transactions")
def get_transactions(
    user_id: str = Query("user_001"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
):
    txns = [t for t in load_transactions() if t["user_id"] == user_id]

    if start_date:
        sd = parse_dt(start_date)
        txns = [t for t in txns if parse_dt(t["timestamp"]) >= sd]
    if end_date:
        ed = parse_dt(end_date)
        txns = [t for t in txns if parse_dt(t["timestamp"]) <= ed]
    if category:
        txns = [t for t in txns if t["category"].lower() == category.lower()]

    return {"user_id": user_id, "count": len(txns), "transactions": txns}


@router.get("/insights")
def get_insights(
    user_id: str = Query("user_001"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    txns = get_transactions(user_id=user_id, start_date=start_date, end_date=end_date)["transactions"]

    total = sum(t["amount"] for t in txns)

    category_totals: dict[str, float] = {}
    for t in txns:
        category_totals[t["category"]] = category_totals.get(t["category"], 0) + t["amount"]

    top_category = max(category_totals, key=lambda k: category_totals[k]) if category_totals else None

    return {
        "user_id": user_id,
        "period": {"start": start_date, "end": end_date},
        "total_spent": round(total, 2),
        "transaction_count": len(txns),
        "top_category": top_category,
        "category_breakdown": {k: round(v, 2) for k, v in sorted(category_totals.items(), key=lambda x: -x[1])},
    }
