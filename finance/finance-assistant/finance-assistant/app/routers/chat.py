import time
from fastapi import APIRouter
from pydantic import BaseModel

from app.agent import get_graph
from app.memory import add_turn, get_history

router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    user_id: str = "user_001"
    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    message: str
    answer: str
    intent: str
    latency_ms: int


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.time()

    history = get_history(req.session_id)

    graph = get_graph()

    result = graph.invoke(
        {
            "user_id": req.user_id,
            "session_id": req.session_id,
            "message": req.message,
            "history": history,
            "intent": "",
            "answer": "",
        }
    )

    # Persist turn
    add_turn(req.session_id, "user", req.message)
    add_turn(req.session_id, "assistant", result["answer"])

    latency = int((time.time() - t0) * 1000)

    return ChatResponse(
        user_id=req.user_id,
        session_id=req.session_id,
        message=req.message,
        answer=result["answer"],
        intent=result["intent"],
        latency_ms=latency,
    )
