"""Ask the Analyst — Step 21's LLM analyst layer: a tool-calling chat
interface grounded in this project's real trained models and reports.

Unlike every other page (which computes its own numbers), this page's
"computation" is delegated to an LLM — so the thing being demonstrated is
that the LLM is FORCED to call real tools (`src/llm/tools.py`) for every
factual claim, not that it's a good conversationalist. Every tool call made
is shown, expanded, alongside the answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.agent import ask_analyst  # noqa: E402
from src.llm.providers import AnalystNotConfiguredError  # noqa: E402

st.set_page_config(page_title="Ask the Analyst", page_icon="💬", layout="wide")
st.title("Ask the Analyst")
st.caption(
    "A tool-calling LLM layer (Step 21): every factual claim below comes from a real tool call "
    'against Steps 10-20\'s actual trained models and computed reports — expand "Tool calls" under '
    "any answer to see exactly what was checked. The one exception is the uplift-modeling tool, "
    "which is always labeled **SIMULATED** since Online Retail II has no real retention campaign."
)

EXAMPLE_QUESTIONS = [
    "What's driving customer 12346's churn risk?",
    "How many customers are high risk and high value?",
    "What's the deployed model's ROC-AUC on the test set?",
    "Is there any data drift right now?",
    "Who are the top 5 customers to prioritize for retention?",
]

if "analyst_messages" not in st.session_state:
    st.session_state.analyst_messages = []

for message in st.session_state.analyst_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_calls"):
            with st.expander(f"Tool calls ({len(message['tool_calls'])})"):
                for call in message["tool_calls"]:
                    st.markdown(f"**`{call['name']}`**({call['arguments']})")
                    st.json(call["result"])

if not st.session_state.analyst_messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS, strict=True):
        col.button(question, key=question, use_container_width=True)

question = st.chat_input("Ask about a customer, the model, drift, or targeting...")
for q in EXAMPLE_QUESTIONS:
    if st.session_state.get(q):
        question = q

if question:
    st.session_state.analyst_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking real tools before answering..."):
                result = ask_analyst(question)
        except AnalystNotConfiguredError:
            st.error(
                "No LLM provider configured. Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env` "
                "to use the analyst layer — see the README's Quick start section."
            )
            st.stop()
        except Exception as exc:  # the provider's own SDK/network errors
            st.error(f"The analyst request failed: {exc}")
            st.stop()

        st.markdown(result.answer)
        tool_calls = [
            {"name": tc.name, "arguments": tc.arguments, "result": tc.result} for tc in result.tool_calls
        ]
        if tool_calls:
            with st.expander(f"Tool calls ({len(tool_calls)})"):
                for call in tool_calls:
                    st.markdown(f"**`{call['name']}`**({call['arguments']})")
                    st.json(call["result"])

    st.session_state.analyst_messages.append(
        {"role": "assistant", "content": result.answer, "tool_calls": tool_calls}
    )
    st.rerun()
