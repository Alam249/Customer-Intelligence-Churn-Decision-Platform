"""The LLM analyst: a grounded, tool-calling question-answering layer over
this project's real models and data (Step 21).

Everything that makes an answer trustworthy lives elsewhere and is reused,
not reimplemented: the tools (`src/llm/tools.py`) wrap Steps 10-20's actual
trained models and computed reports; the provider loop (`src/llm/providers.py`)
executes them exactly as either LLM API requests. This module is just the
system prompt that tells the model to always use them, plus the one function
(`ask_analyst`) both the API endpoint and the dashboard chat page call.
"""

from __future__ import annotations

from src.llm.providers import AgentResult, get_provider
from src.llm.tools import TOOLS

ANALYST_SYSTEM_PROMPT = """\
You are a data analyst assistant for a customer churn prediction platform built on the \
Online Retail II dataset (a UK online gift retailer). You have tools that query REAL, \
live-computed data from trained models and the actual customer population.

Rules, in order of importance:
1. Before making ANY factual claim about a specific customer, the model's performance, \
data drift, or the uplift analysis, call the relevant tool. Never state a number you did \
not get from a tool result in this conversation, even if you believe you already know it.
2. If no available tool can answer the question, say so plainly rather than guessing or \
using general knowledge about churn or retail.
3. If a tool result contains "SIMULATED": true (the uplift-modeling tool), you MUST tell \
the user these are simulated/synthetic numbers built for methodology demonstration, not a \
real measured outcome — never present them as if they describe real customer behaviour.
4. If a tool result contains an "error" key, relay that error plainly to the user rather \
than inventing a workaround or a plausible-sounding answer.
5. Be concise and precise. When you cite a number, mention which check it came from (e.g. \
"the deployed model's test-set ROC-AUC is 0.81").
"""


def ask_analyst(question: str) -> AgentResult:
    """Ask the configured LLM provider a question; it will call whichever
    real tools it needs before answering. Raises
    `src.llm.providers.AnalystNotConfiguredError` if no API key is set.
    """
    provider = get_provider()
    return provider.ask(ANALYST_SYSTEM_PROMPT, question, TOOLS)
