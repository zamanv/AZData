"""Shared pytest fixtures for the AZData test suite.

Adds the repository root to ``sys.path`` so the flat top-level packages
(``analytics``, ``ingestion``, ``llm``, ``profiling``, ``reports``,
``security``, ``utils``) import cleanly without an install step.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── deterministic random state for every test ──
@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ── a frame with exactly one extreme outlier in column "v" ──
@pytest.fixture
def numeric_df(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "v": list(rng.normal(0, 1, 100)) + [50.0],
            "w": rng.normal(10, 2, 101).tolist(),
        }
    )


# ── a heterogeneous frame exercising every type-inference branch ──
@pytest.fixture
def mixed_df(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "v": list(rng.normal(0, 1, 100)) + [50.0],
            "w": rng.normal(10, 2, 101).tolist(),
            "cat": ["x", "y", "z"] * 33 + ["x", "y"],
            "flag": [True] * 51 + [False] * 50,
            "created_at": pd.date_range("2024-01-01", periods=101, freq="D").tolist(),
        }
    )


# ── clean frame: no missing values, no duplicates, no constant columns ──
@pytest.fixture
def clean_df(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, 60),
            "b": ["p", "q"] * 30,
        }
    )


# ── deliberately broken frame: missing values, duplicates, constant col ──
@pytest.fixture
def dirty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, None, None, None, 4, 5, 6, 7, 8, 9],
            "b": [1] * 10,
        }
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
