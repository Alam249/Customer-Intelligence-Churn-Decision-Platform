"""POST /analyst/ask — the LLM analyst layer (Step 21).

Every factual claim in the answer is grounded in a real tool call against
Steps 10-20's actual trained models and computed reports (`src/llm/tools.py`)
— never the LLM's own general knowledge. `tool_calls` in the response is the
full, auditable trace of what was actually checked, not a black box.

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY to be set (see .env.example);
returns 503 rather than crashing if neither is configured.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.analyst import AskRequest, AskResponse
from src.llm.agent import ask_analyst
from src.llm.providers import AnalystNotConfiguredError
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analyst", tags=["analyst"])


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        503: {"description": "No LLM provider configured (set ANTHROPIC_API_KEY or OPENAI_API_KEY)."},
        500: {"description": "The LLM provider call failed. See server logs."},
    },
)
def ask(request: AskRequest) -> AskResponse:
    """Ask the analyst a question; it calls whichever real tools it needs
    before answering. May take up to ~30s if the (simulated) uplift analysis
    is involved and hasn't been computed yet this process.
    """
    try:
        result = ask_analyst(request.question)
    except AnalystNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analyst request failed for question: %s", request.question)
        raise HTTPException(status_code=500, detail="Analyst request failed. See server logs.") from exc

    return AskResponse(
        answer=result.answer,
        tool_calls=[
            {"name": tc.name, "arguments": tc.arguments, "result": tc.result} for tc in result.tool_calls
        ],
        provider=result.provider,
        model=result.model,
    )
