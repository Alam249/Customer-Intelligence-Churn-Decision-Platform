"""Pydantic request/response schemas for the LLM analyst endpoint (Step 21)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="A natural-language question about a customer, the model, drift, or the "
        "(simulated) uplift analysis.",
        examples=["What's driving customer 12346's churn risk?"],
    )


class ToolCallRecord(BaseModel):
    name: str = Field(..., description="Which real tool/computation was called.")
    arguments: dict[str, Any] = Field(..., description="Arguments the model passed to the tool.")
    result: dict[str, Any] = Field(..., description="The tool's real, live-computed result.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The analyst's answer, grounded in the tool calls below.")
    tool_calls: list[ToolCallRecord] = Field(
        ..., description="Every real tool call made while answering — full auditability, not a black box."
    )
    provider: str = Field(..., description="Which LLM provider answered ('anthropic' or 'openai').")
    model: str = Field(..., description="Which model answered.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Customer 12346 has a 37.4% churn probability (Medium risk). The top factor "
                "increasing risk is a low purchase_rate_per_month; the top protective factor is a "
                "high rfm_score.",
                "tool_calls": [
                    {
                        "name": "explain_customer_churn",
                        "arguments": {"customer_id": 12346},
                        "result": {"customer_id": 12346, "churn_probability": 0.3736},
                    }
                ],
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
            }
        }
    }
