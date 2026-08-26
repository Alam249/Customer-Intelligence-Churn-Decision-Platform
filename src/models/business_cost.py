"""A simple business-cost framework for choosing a decision threshold.

IMPORTANT — what is real vs. hypothetical in this module
----------------------------------------------------------
``value_per_customer`` is measured from actual data (median `monetary_total`
on the training split — see the calling script). Everything else —
``contact_cost`` and ``retention_success_rate`` — is a labelled, hypothetical
assumption for demonstration purposes, because Online Retail II contains no
record of any past retention campaign or its cost or success rate. Real
values would come from a company's own outreach program; the point of this
module is the FRAMEWORK (how a threshold decision follows from cost
assumptions), not a specific number this project could not honestly know.

Framework
---------
Assigns a monetary cost to each confusion-matrix outcome at a given threshold
(the classic cost-sensitive-learning setup):

    cost(TP) = contact_cost - retention_success_rate * value_per_customer
               (usually negative -> a net benefit: contacting a real churner
                and sometimes saving them is worth more than the outreach cost)
    cost(FP) = contact_cost
               (wasted outreach on a customer who was never leaving)
    cost(FN) = retention_success_rate * value_per_customer
               (opportunity cost: the value that COULD have been saved by an
                offer, forgone because the customer wasn't flagged)
    cost(TN) = 0

The optimal threshold minimises total cost across the test set — equivalently,
maximises net value versus doing nothing at all.

Uplift caveat this framework does NOT capture: it assumes
``retention_success_rate`` is the same for every flagged churner. In reality
some customers would have stayed anyway (contacting them is pure waste beyond
the FP cost already charged) and some can't be saved by any offer — that
heterogeneity is exactly what Step 20's uplift modelling addresses. This
framework is a legitimate first pass, not a substitute for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BusinessCostAssumptions:
    contact_cost: float
    value_per_customer: float
    retention_success_rate: float
    label: str = "scenario"

    @property
    def cost_tp(self) -> float:
        return self.contact_cost - self.retention_success_rate * self.value_per_customer

    @property
    def cost_fp(self) -> float:
        return self.contact_cost

    @property
    def cost_fn(self) -> float:
        return self.retention_success_rate * self.value_per_customer

    cost_tn: float = 0.0


def cost_at_threshold(y_true, y_proba, threshold: float, assumptions: BusinessCostAssumptions) -> dict:
    """Confusion-matrix counts and total cost for one threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_proba) >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    total_cost = (
        tp * assumptions.cost_tp + fp * assumptions.cost_fp
        + fn * assumptions.cost_fn + tn * assumptions.cost_tn
    )
    # Net value relative to a "do nothing, contact no one" policy, whose cost
    # is n_actual_positives * cost_fn (everyone who churns is a missed
    # opportunity) — this is what makes "net value" interpretable as a gain.
    do_nothing_cost = (tp + fn) * assumptions.cost_fn
    net_value_vs_doing_nothing = do_nothing_cost - total_cost

    return {
        "threshold": round(threshold, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_cost": round(total_cost, 2),
        "net_value_vs_doing_nothing": round(net_value_vs_doing_nothing, 2),
    }


def sweep_thresholds(y_true, y_proba, assumptions: BusinessCostAssumptions, thresholds=None) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.arange(0.05, 0.96, 0.01)
    return pd.DataFrame([cost_at_threshold(y_true, y_proba, t, assumptions) for t in thresholds])


def find_optimal_threshold(sweep_table: pd.DataFrame) -> pd.Series:
    """The threshold minimising total cost (== maximising net value)."""
    return sweep_table.loc[sweep_table["total_cost"].idxmin()]
