"""Tiny shared helper for turning a DataFrame into GitHub-flavoured Markdown.

Used by every ``scripts/run_*.py`` report generator so there is one
implementation instead of one per script.
"""

from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame as a Markdown table without extra dependencies."""
    d = df.reset_index() if index else df
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in d.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])
