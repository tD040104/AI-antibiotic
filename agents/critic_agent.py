"""
Critic agent - detect uncertain predictions or missing data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class CriticFlag:
    antibiotic: str
    probability: float
    reason: str


class CriticAgent:
    """
    Agent 4: flag low-confidence predictions or incomplete patient data.
    """

    def __init__(self, uncertainty_band: Tuple[float, float] = (0.4, 0.6)):
        self.uncertainty_band = uncertainty_band

    def review(
        self,
        probabilities: pd.DataFrame,
        patient_features: Optional[pd.Series] = None,
        missing_required_fields: Optional[List[str]] = None,
    ) -> Dict:
        if probabilities.empty:
            return {"flags": [], "missing_fields": missing_required_fields or []}

        lower, upper = self.uncertainty_band
        proba_row = probabilities.iloc[0]

        flags: List[CriticFlag] = []
        for antibiotic, proba in proba_row.items():
            if lower <= proba <= upper:
                flags.append(CriticFlag(antibiotic, float(proba), "uncertain_probability"))

        missing = missing_required_fields or []
        if patient_features is not None:
            for col, value in patient_features.items():
                if pd.isna(value) or value in ["", "Unknown"]:
                    missing.append(col)

        return {
            "flags": flags,
            "missing_fields": sorted(set(missing)),
            "needs_additional_tests": bool(flags or missing),
        }

