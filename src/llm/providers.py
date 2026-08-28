"""Provider-agnostic LLM client for the analyst layer (Step 21).

Two providers, one hand-rolled tool-use loop each — no LangChain or similar
framework. Anthropic's and OpenAI's native tool-calling protocols differ
enough (message/content-block shape, how a tool result is fed back) that a
thin per-provider implementation is clearer and more auditable than forcing
both through one framework's lowest-common-denominator abstraction — the
same hand-rolled-over-framework choice this project already made for
monitoring (Step 19, over Evidently) and uplift modeling (Step 20, over
causalml/econml).

Provider selection: `ANTHROPIC_API_KEY` takes priority if both are set (this
project was built with Claude Code); otherwise `OPENAI_API_KEY` is used.
Model names are configurable in `config/config.yaml` under `llm:`, never
hardcoded here — set neither and `get_provider()` raises a clear error
rather than silently doing nothing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.config import CONFIG
from src.llm.tools import Tool
from src.utils.logging import get_logger

logger = get_logger(__name__)

MAX_TOOL_ITERATIONS = 6
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class AnalystNotConfiguredError(RuntimeError):
    """Raised when neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set."""


@dataclass
class ToolCallRecord:
    """One executed tool call, kept for transparency/auditability — the API
    and dashboard both surface this alongside the final answer so a user
    can see exactly which real computation backs each claim.
    """

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    provider: str = ""
    model: str = ""


class Analyst(Protocol):
    def ask(self, system_prompt: str, question: str, tools: list[Tool]) -> AgentResult: ...


def _execute_tool(tools: list[Tool], name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one tool call. Returns an `{"error": ...}` dict rather than
    raising — a broken tool call must not crash the whole conversation, it
    should give the model something it can react to (e.g. re-asking the
    user for a valid input).
    """
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return tool.function(**arguments)
    except Exception as exc:
        logger.exception("Tool '%s' raised an exception with arguments %s", name, arguments)
        return {"error": f"Tool '{name}' failed: {exc}"}


class AnthropicAnalyst:
    """Tool-use loop against Claude's Messages API."""

    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self.client = client
        self.model = model

    def ask(self, system_prompt: str, question: str, tools: list[Tool]) -> AgentResult:
        tool_specs = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        tool_calls: list[ToolCallRecord] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=tool_specs,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                text = "".join(block.text for block in response.content if block.type == "text")
                return AgentResult(answer=text, tool_calls=tool_calls, provider="anthropic", model=self.model)

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                arguments = dict(block.input)
                result = _execute_tool(tools, block.name, arguments)
                tool_calls.append(ToolCallRecord(name=block.name, arguments=arguments, result=result))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
                )
            messages.append({"role": "user", "content": tool_results})

        return AgentResult(
            answer="I wasn't able to reach a final answer within the allotted tool-call budget.",
            tool_calls=tool_calls,
            provider="anthropic",
            model=self.model,
        )


class OpenAIAnalyst:
    """Tool-use loop against OpenAI's Chat Completions API."""

    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        if client is None:
            import openai

            client = openai.OpenAI(api_key=api_key)
        self.client = client
        self.model = model

    def ask(self, system_prompt: str, question: str, tools: list[Tool]) -> AgentResult:
        tool_specs = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        tool_calls: list[ToolCallRecord] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tool_specs
            )
            message = response.choices[0].message

            # Built explicitly from the fields actually used below, rather
            # than a blind `message.model_dump()` — keeps the exact wire
            # shape this loop depends on visible and independently testable
            # with a plain mock instead of a real pydantic response object.
            assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ]
            messages.append(assistant_message)

            if not message.tool_calls:
                return AgentResult(
                    answer=message.content or "", tool_calls=tool_calls, provider="openai", model=self.model
                )

            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                result = _execute_tool(tools, call.function.name, arguments)
                tool_calls.append(ToolCallRecord(name=call.function.name, arguments=arguments, result=result))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

        return AgentResult(
            answer="I wasn't able to reach a final answer within the allotted tool-call budget.",
            tool_calls=tool_calls,
            provider="openai",
            model=self.model,
        )


def get_provider() -> Analyst:
    """`ANTHROPIC_API_KEY` wins if both are set; otherwise `OPENAI_API_KEY`.
    Raises `AnalystNotConfiguredError` if neither is set, rather than
    returning something that would fail confusingly on first use.
    """
    llm_cfg = CONFIG.get("llm", {})
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        return AnthropicAnalyst(
            api_key=anthropic_key, model=llm_cfg.get("anthropic_model", DEFAULT_ANTHROPIC_MODEL)
        )
    if openai_key:
        return OpenAIAnalyst(api_key=openai_key, model=llm_cfg.get("openai_model", DEFAULT_OPENAI_MODEL))
    raise AnalystNotConfiguredError(
        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env to use the LLM analyst layer (Step 21)."
    )
