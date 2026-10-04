"""Deterministic synthetic retail dataset with planted phenomena (docs/10_verification.md §10.2).

Mechanisms (global):
    margin      = 26 - 0.6 * discount + N(0, 3)          (discount erodes margin)
    return_rate = 0.02 + 0.01 * delivery_days + N(0, .008) (delays drive returns)

Planted subgroup effects (the ground truth the E2E tests look for):
    local anomaly          EU & laptops                 margin +5
    stronger specialization EU & laptops & online        margin +4 more (+9 total)
    contrasting subgroup   EU & laptops & retail        margin -9 (net -4 vs global)
    recurring phenomenon   US & phones, APAC & tablets,  discount +9  => margin down
      "discount erosion"   EU & tablets & retail         (structurally disjoint scopes)
    recurring phenomenon   APAC & online,                delivery +3 days => returns up
      "delay -> returns"   US & laptops & retail
    correlation break      EU & phones                   margin decoupled from discount (EMM)

Columns payment / weekday / store_size carry no signal; order_id is an identifier.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUND_TRUTH = {
    "local_anomaly": {"scope": {"region": "EU", "category": "laptops"}, "metric": "margin", "direction": 1},
    "stronger_specialization": {
        "scope": {"region": "EU", "category": "laptops", "channel": "online"},
        "metric": "margin",
        "direction": 1,
        "parent": "local_anomaly",
    },
    "contrasting_subgroup": {
        "scope": {"region": "EU", "category": "laptops", "channel": "retail"},
        "metric": "margin",
        "direction": -1,
        "contrasts": "local_anomaly",
    },
    "discount_erosion": {
        "scopes": [
            {"region": "US", "category": "phones"},
            {"region": "APAC", "category": "tablets"},
            {"region": "EU", "category": "tablets", "channel": "retail"},
        ],
        "metrics": {"discount": 1, "margin": -1},
    },
    "delay_returns": {
        "scopes": [
            {"region": "APAC", "channel": "online"},
            {"region": "US", "category": "laptops", "channel": "retail"},
        ],
        "metrics": {"delivery_days": 1, "return_rate": 1},
    },
    "correlation_break": {"scope": {"region": "EU", "category": "phones"}},
}


def _choice(rng: np.random.RandomState, values: list[str], probs: list[float], n: int) -> np.ndarray:
    return rng.choice(np.array(values, dtype=object), size=n, p=probs)


def generate_retail_dataset(n_rows: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    region = _choice(rng, ["EU", "US", "APAC", "LATAM"], [0.35, 0.30, 0.25, 0.10], n_rows)
    category = _choice(rng, ["laptops", "phones", "tablets", "accessories"], [0.30, 0.30, 0.22, 0.18], n_rows)
    channel = _choice(rng, ["online", "retail", "partner"], [0.50, 0.35, 0.15], n_rows)
    payment = _choice(rng, ["card", "cash", "wallet"], [0.5, 0.3, 0.2], n_rows)
    weekday = _choice(rng, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], [1 / 7] * 7, n_rows)
    store_size = _choice(rng, ["S", "M", "L"], [0.3, 0.4, 0.3], n_rows)

    def mask(**scope: str) -> np.ndarray:
        m = np.ones(n_rows, dtype=bool)
        cols = {"region": region, "category": category, "channel": channel}
        for key, value in scope.items():
            m &= cols[key] == value
        return m

    discount = np.clip(rng.normal(10.0, 3.0, n_rows), 0.0, 40.0)
    for scope in GROUND_TRUTH["discount_erosion"]["scopes"]:
        discount[mask(**scope)] += 9.0

    margin = 26.0 - 0.6 * discount + rng.normal(0.0, 3.0, n_rows)
    broken = mask(**GROUND_TRUTH["correlation_break"]["scope"])
    margin[broken] = 26.0 - 0.6 * 10.0 + rng.normal(0.0, 3.0, int(broken.sum()))
    margin[mask(region="EU", category="laptops")] += 5.0
    margin[mask(region="EU", category="laptops", channel="online")] += 4.0
    margin[mask(region="EU", category="laptops", channel="retail")] -= 9.0

    delivery = 2.0 + rng.gamma(2.0, 0.7, n_rows)
    for scope in GROUND_TRUTH["delay_returns"]["scopes"]:
        delivery[mask(**scope)] += 3.0
    return_rate = np.clip(0.02 + 0.01 * delivery + rng.normal(0.0, 0.008, n_rows), 0.0, 1.0)

    return pd.DataFrame(
        {
            "order_id": np.arange(100000, 100000 + n_rows),
            "region": region.astype(str),
            "category": category.astype(str),
            "channel": channel.astype(str),
            "payment": payment.astype(str),
            "weekday": weekday.astype(str),
            "store_size": store_size.astype(str),
            "discount": discount.round(2),
            "margin": margin.round(2),
            "delivery_days": delivery.round(1),
            "return_rate": return_rate.round(4),
        }
    )
