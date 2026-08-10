"""Shared helpers for the marimo analysis notebooks. Thin wrappers over results/*.csv."""
import os
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")

# display order high -> low signal (matches src/cohorts.HIERARCHY)
ORDER = ["tcrvdb_true", "vdjdb_hq", "immrep22_true", "vdjdb_lq", "tcrvdb_false",
         "immrep25_pos", "airr_top", "mlr_prolif", "airr_control", "olga_random"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, name))


def ordered(df: pd.DataFrame, col: str = "cohort", order=None) -> pd.DataFrame:
    order = order or ORDER
    key = {c: i for i, c in enumerate(order)}
    return df.assign(_k=df[col].map(lambda c: key.get(c, 99))).sort_values("_k").drop(columns="_k")
