"""R1A inventory locks: sample size and effective parameter counts with sigma_sys."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ptquat.data import load_tidy_sparc
from ptquat.fit_global import _layout

IMPL_ROOT = Path(__file__).resolve().parents[1]
TIDY = IMPL_ROOT / "dataset" / "sparc_tidy.csv"


def test_sparc_tidy_sample_size():
    df = pd.read_csv(TIDY)
    assert df["galaxy"].nunique() == 91
    assert len(df) == 1782
    gals = load_tidy_sparc(str(TIDY))
    assert len(gals) == 91
    assert sum(len(g.r_kpc) for g in gals.values()) == 1782


def test_effective_parameter_counts_with_learned_sigma_sys():
    G = 91
    expected = {
        "baryon": 92,
        "mond": 93,
        "ptq": 93,
        "ptq-nu": 93,
        "ptq-screen": 94,
        "mond-screen": 94,
        "nfw1p": 183,
    }
    for model, k in expected.items():
        L = _layout(model, G, sigma_sys_learn=True, a0_fixed=None)
        assert L["k"] == k, f"{model}: got {L['k']}, expected {k}"
        assert "i_lnsig" in L
