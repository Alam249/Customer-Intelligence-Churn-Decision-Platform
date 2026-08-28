"""Tests for src/llm/providers.py (Step 21's tool-calling loop).

No real API calls here — both `AnthropicAnalyst` and `OpenAIAnalyst` accept
an injectable `client`, so the loop's actual logic (does it call the right
tool, feed the result back in the shape each API expects, stop when the
model gives a final answer, and not loop forever) is tested against small
fake objects that mimic just the attributes each provider's code reads from
a real SDK response — deterministic, free, and independently verified
against the real SDKs' response shapes (`anthropic.types.Message`/
`ToolUseBlock`, `openai.types.chat.ChatCompletionMessage`) while building this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.llm.providers import (
    MAX_TOOL_ITERATIONS,
    AnalystNotConfiguredError,
    AnthropicAnalyst,
    OpenAIAnalyst,
    get_provider,
)
from src.llm.tools import Tool

# ---------------------------------------------------------------------------
# A trivial real (not mocked) tool, used by every loop test.
# ---------------------------------------------------------------------------

DUMMY_TOOL = Tool(
    name="dummy_tool",
    description="A test tool that doubles a number.",
    parameters={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
    function=lambda x: {"doubled": x * 2},
)

BROKEN_TOOL = Tool(
    name="broken_tool",
    description="A test tool that always raises.",
    parameters={"type": "object", "properties": {}},
    function=lambda: (_ for _ in ()).throw(ValueError("boom")),
)


# ---------------------------------------------------------------------------
# Anthropic fakes — mimic anthropic.types.Message / TextBlock / ToolUseBlock
# ---------------------------------------------------------------------------


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeAnthropicMessage:
    content: list[Any]
    stop_reason: str


class FakeAnthropicClient:
    """Returns pre-scripted responses in order; records every call's kwargs."""

    def __init__(self, responses: list[FakeAnthropicMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self  # so self.client.messages.create(...) resolves

    def create(self, **kwargs: Any) -> FakeAnthropicMessage:
        # Snapshot `messages` (a list `providers.py` keeps mutating in place
        # after this call returns) so a later assertion sees this call's
        # state at the time it was made, not whatever the list grows into.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self._responses:
            raise AssertionError("FakeAnthropicClient ran out of scripted responses")
        return self._responses.pop(0)


def test_anthropic_analyst_immediate_text_answer_makes_no_tool_calls():
    client = FakeAnthropicClient(
        [FakeAnthropicMessage(content=[FakeTextBlock("The answer is 4.")], stop_reason="end_turn")]
    )
    analyst = AnthropicAnalyst(api_key="dummy", model="test-model", client=client)

    result = analyst.ask("system prompt", "what is 2+2?", [DUMMY_TOOL])

    assert result.answer == "The answer is 4."
    assert result.tool_calls == []
    assert len(client.calls) == 1
    assert client.calls[0]["system"] == "system prompt"
    assert client.calls[0]["tools"][0]["name"] == "dummy_tool"
    assert client.calls[0]["tools"][0]["input_schema"] == DUMMY_TOOL.parameters


def test_anthropic_analyst_executes_tool_then_returns_final_answer():
    client = FakeAnthropicClient(
        [
            FakeAnthropicMessage(
                content=[FakeToolUseBlock(id="call_1", name="dummy_tool", input={"x": 21})],
                stop_reason="tool_use",
            ),
            FakeAnthropicMessage(content=[FakeTextBlock("21 doubled is 42.")], stop_reason="end_turn"),
        ]
    )
    analyst = AnthropicAnalyst(api_key="dummy", model="test-model", client=client)

    result = analyst.ask("system prompt", "what is 21 doubled?", [DUMMY_TOOL])

    assert result.answer == "21 doubled is 42."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "dummy_tool"
    assert result.tool_calls[0].arguments == {"x": 21}
    assert result.tool_calls[0].result == {"doubled": 42}

    # Second API call must carry the tool result back, addressed to the
    # right tool_use_id — the exact thing the real API validates.
    second_call_messages = client.calls[1]["messages"]
    tool_result_message = second_call_messages[-1]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["tool_use_id"] == "call_1"
    assert json.loads(tool_result_message["content"][0]["content"]) == {"doubled": 42}


def test_anthropic_analyst_broken_tool_returns_error_without_crashing():
    client = FakeAnthropicClient(
        [
            FakeAnthropicMessage(
                content=[FakeToolUseBlock(id="call_1", name="broken_tool", input={})], stop_reason="tool_use"
            ),
            FakeAnthropicMessage(content=[FakeTextBlock("Something went wrong.")], stop_reason="end_turn"),
        ]
    )
    analyst = AnthropicAnalyst(api_key="dummy", model="test-model", client=client)

    result = analyst.ask("system prompt", "trigger the broken tool", [BROKEN_TOOL])

    assert "error" in result.tool_calls[0].result
    assert "boom" in result.tool_calls[0].result["error"]
    assert result.answer == "Something went wrong."


def test_anthropic_analyst_unknown_tool_name_returns_error():
    client = FakeAnthropicClient(
        [
            FakeAnthropicMessage(
                content=[FakeToolUseBlock(id="call_1", name="does_not_exist", input={})],
                stop_reason="tool_use",
            ),
            FakeAnthropicMessage(content=[FakeTextBlock("ok")], stop_reason="end_turn"),
        ]
    )
    analyst = AnthropicAnalyst(api_key="dummy", model="test-model", client=client)
    result = analyst.ask("system prompt", "question", [DUMMY_TOOL])
    assert "Unknown tool" in result.tool_calls[0].result["error"]


def test_anthropic_analyst_stops_after_max_iterations_instead_of_looping_forever():
    always_tool_use = FakeAnthropicMessage(
        content=[FakeToolUseBlock(id="call_x", name="dummy_tool", input={"x": 1})], stop_reason="tool_use"
    )
    client = FakeAnthropicClient([always_tool_use] * MAX_TOOL_ITERATIONS)
    analyst = AnthropicAnalyst(api_key="dummy", model="test-model", client=client)

    result = analyst.ask("system prompt", "loop forever", [DUMMY_TOOL])

    assert len(client.calls) == MAX_TOOL_ITERATIONS
    assert "tool-call budget" in result.answer
    assert len(result.tool_calls) == MAX_TOOL_ITERATIONS


# ---------------------------------------------------------------------------
# OpenAI fakes — mimic ChatCompletion / ChatCompletionMessage / tool_calls
# ---------------------------------------------------------------------------


@dataclass
class FakeFunctionCall:
    name: str
    arguments: str  # JSON-encoded string, matching the real API


@dataclass
class FakeOpenAIToolCall:
    id: str
    function: FakeFunctionCall


@dataclass
class FakeOpenAIMessage:
    content: str | None
    tool_calls: list[FakeOpenAIToolCall] = field(default_factory=list)


@dataclass
class FakeChoice:
    message: FakeOpenAIMessage


@dataclass
class FakeChatCompletion:
    choices: list[FakeChoice]


class FakeOpenAIClient:
    def __init__(self, responses: list[FakeChatCompletion]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> FakeChatCompletion:
        # See FakeAnthropicClient.create — snapshot messages before providers.py
        # keeps appending to the same list after this call returns.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self._responses:
            raise AssertionError("FakeOpenAIClient ran out of scripted responses")
        return self._responses.pop(0)


def test_openai_analyst_immediate_text_answer_makes_no_tool_calls():
    client = FakeOpenAIClient(
        [FakeChatCompletion(choices=[FakeChoice(FakeOpenAIMessage(content="The answer is 4."))])]
    )
    analyst = OpenAIAnalyst(api_key="dummy", model="test-model", client=client)

    result = analyst.ask("system prompt", "what is 2+2?", [DUMMY_TOOL])

    assert result.answer == "The answer is 4."
    assert result.tool_calls == []
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "system prompt"}
    assert client.calls[0]["tools"][0]["function"]["name"] == "dummy_tool"


def test_openai_analyst_executes_tool_then_returns_final_answer():
    client = FakeOpenAIClient(
        [
            FakeChatCompletion(
                choices=[
                    FakeChoice(
                        FakeOpenAIMessage(
                            content=None,
                            tool_calls=[
                                FakeOpenAIToolCall(
                                    id="call_1",
                                    function=FakeFunctionCall("dummy_tool", json.dumps({"x": 21})),
                                )
                            ],
                        )
                    )
                ]
            ),
            FakeChatCompletion(choices=[FakeChoice(FakeOpenAIMessage(content="21 doubled is 42."))]),
        ]
    )
    analyst = OpenAIAnalyst(api_key="dummy", model="test-model", client=client)

    result = analyst.ask("system prompt", "what is 21 doubled?", [DUMMY_TOOL])

    assert result.answer == "21 doubled is 42."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].arguments == {"x": 21}
    assert result.tool_calls[0].result == {"doubled": 42}

    tool_response_message = client.calls[1]["messages"][-1]
    assert tool_response_message == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps({"doubled": 42}),
    }


def test_openai_analyst_broken_tool_returns_error_without_crashing():
    client = FakeOpenAIClient(
        [
            FakeChatCompletion(
                choices=[
                    FakeChoice(
                        FakeOpenAIMessage(
                            content=None,
                            tool_calls=[
                                FakeOpenAIToolCall(
                                    id="call_1", function=FakeFunctionCall("broken_tool", "{}")
                                )
                            ],
                        )
                    )
                ]
            ),
            FakeChatCompletion(choices=[FakeChoice(FakeOpenAIMessage(content="Something went wrong."))]),
        ]
    )
    analyst = OpenAIAnalyst(api_key="dummy", model="test-model", client=client)
    result = analyst.ask("system prompt", "trigger the broken tool", [BROKEN_TOOL])
    assert "boom" in result.tool_calls[0].result["error"]


def test_openai_analyst_stops_after_max_iterations_instead_of_looping_forever():
    always_tool_call = FakeChatCompletion(
        choices=[
            FakeChoice(
                FakeOpenAIMessage(
                    content=None,
                    tool_calls=[
                        FakeOpenAIToolCall(
                            id="call_x", function=FakeFunctionCall("dummy_tool", json.dumps({"x": 1}))
                        )
                    ],
                )
            )
        ]
    )
    client = FakeOpenAIClient([always_tool_call] * MAX_TOOL_ITERATIONS)
    analyst = OpenAIAnalyst(api_key="dummy", model="test-model", client=client)

    result = analyst.ask("system prompt", "loop forever", [DUMMY_TOOL])

    assert len(client.calls) == MAX_TOOL_ITERATIONS
    assert "tool-call budget" in result.answer


# ---------------------------------------------------------------------------
# get_provider() selection logic
# ---------------------------------------------------------------------------


def test_get_provider_raises_when_neither_key_is_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AnalystNotConfiguredError):
        get_provider()


def test_get_provider_prefers_anthropic_when_both_keys_are_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    assert isinstance(get_provider(), AnthropicAnalyst)


def test_get_provider_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    assert isinstance(get_provider(), OpenAIAnalyst)
