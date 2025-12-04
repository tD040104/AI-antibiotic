"""
Decision agent - combine critic insights + treatment recommendations
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


class TreatmentRecommenderAgent:
    """Domain rules to rank antibiotics based on sensitivity probability."""

    def __init__(self):
        self.antibiotic_names = {
            "AMX/AMP": "Amoxicillin/Ampicillin",
            "AMC": "Amoxicillin-Clavulanic Acid",
            "CZ": "Cefazolin",
            "FOX": "Cefoxitin",
            "CTX/CRO": "Ceftriaxone/Cefotaxime",
            "IPM": "Imipenem",
            "GEN": "Gentamicin",
            "AN": "Amikacin",
            "Acide nalidixique": "Nalidixic Acid",
            "ofx": "Ofloxacin",
            "CIP": "Ciprofloxacin",
            "C": "Chloramphenicol",
            "Co-trimoxazole": "Trimethoprim-Sulfamethoxazole",
            "Furanes": "Nitrofurantoin",
            "colistine": "Colistin",
        }

        self.antibiotic_groups = {
            "Beta-lactams": ["AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO"],
            "Carbapenems": ["IPM"],
            "Aminoglycosides": ["GEN", "AN"],
            "Quinolones": ["Acide nalidixique", "ofx", "CIP"],
            "Others": ["C", "Co-trimoxazole", "Furanes", "colistine"],
        }

        self.antibiotic_priority = {
            "IPM": 1,
            "AN": 2,
            "GEN": 3,
            "CIP": 4,
            "ofx": 5,
            "CTX/CRO": 6,
            "AMC": 7,
            "CZ": 8,
            "FOX": 9,
            "AMX/AMP": 10,
            "colistine": 1,
            "Co-trimoxazole": 11,
            "Acide nalidixique": 12,
            "C": 13,
            "Furanes": 14,
        }

    def recommend_treatment(
        self,
        resistance_proba: pd.DataFrame,
        patient_data: Optional[pd.Series] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        recommendations: List[Dict] = []
        if len(resistance_proba) == 0:
            return recommendations
        proba = resistance_proba.iloc[0]
        sorted_antibiotics = proba.sort_values(ascending=False)
        sensitive_antibiotics = sorted_antibiotics[sorted_antibiotics >= 0.5]
        if len(sensitive_antibiotics) == 0:
            sensitive_antibiotics = sorted_antibiotics.head(top_k)

        recommendations = self._apply_medical_guidelines(
            sensitive_antibiotics, patient_data, top_k
        )
        return recommendations

    def _apply_medical_guidelines(
        self,
        sensitive_proba: pd.Series,
        patient_data: Optional[pd.Series] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        candidates = []
        for antibiotic, proba in sensitive_proba.items():
            score = proba
            if antibiotic in self.antibiotic_priority:
                priority_bonus = 1.0 / (self.antibiotic_priority[antibiotic] + 1)
                score += priority_bonus * 0.2

            if patient_data is not None:
                if antibiotic == "IPM" and patient_data.get("Total_risk_factors", 0) < 2:
                    score *= 0.8
                if antibiotic in ["GEN", "AN"] and patient_data.get("Total_risk_factors", 0) >= 2:
                    score *= 1.1

            candidates.append({
                "antibiotic": antibiotic,
                "full_name": self.antibiotic_names.get(antibiotic, antibiotic),
                "sensitive_probability": proba,
                "score": score,
                "group": self._get_antibiotic_group(antibiotic),
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = []
        groups_used = set()
        for candidate in candidates:
            if len(selected) >= top_k:
                break
            if candidate["group"] not in groups_used or len(groups_used) >= top_k:
                selected.append(candidate)
                groups_used.add(candidate["group"])

        while len(selected) < top_k and len(selected) < len(candidates):
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                    break

        recommendations = []
        for idx, candidate in enumerate(selected[:top_k]):
            recommendations.append({
                "rank": idx + 1,
                "antibiotic_code": candidate["antibiotic"],
                "antibiotic_name": candidate["full_name"],
                "sensitive_probability": round(candidate["sensitive_probability"], 3),
                "confidence": self._get_confidence_level(candidate["sensitive_probability"]),
                "group": candidate["group"],
                "recommendation": self._generate_recommendation_text(candidate),
            })
        return recommendations

    def _get_antibiotic_group(self, antibiotic: str) -> str:
        for group, antibiotics in self.antibiotic_groups.items():
            if antibiotic in antibiotics:
                return group
        return "Others"

    def _get_confidence_level(self, proba: float) -> str:
        if proba >= 0.8:
            return "High"
        if proba >= 0.6:
            return "Medium"
        return "Low"

    def _generate_recommendation_text(self, candidate: Dict) -> str:
        proba = candidate["sensitive_probability"]
        if proba >= 0.8:
            return "Highly recommended - High sensitivity probability"
        if proba >= 0.6:
            return "Recommended - Moderate sensitivity probability"
        return "Consider with caution - Lower sensitivity probability"


class DecisionAgent:
    """
    Agent 5: combine critic feedback + treatment ranking to output actions.
    """

    def __init__(self, treatment_agent: Optional[TreatmentRecommenderAgent] = None):
        self.treatment_agent = treatment_agent or TreatmentRecommenderAgent()

    def decide(
        self,
        probabilities: pd.DataFrame,
        critic_report: Dict,
        patient_features: Optional[pd.Series] = None,
        top_k: int = 3,
    ) -> Dict:
        recommendations = self.treatment_agent.recommend_treatment(
            probabilities,
            patient_features,
            top_k=top_k,
        )

        actions = []
        if critic_report.get("needs_additional_tests"):
            if critic_report.get("flags"):
                flagged_codes = [flag.antibiotic for flag in critic_report["flags"]]
                actions.append(
                    f"Yêu cầu xét nghiệm bổ sung cho: {', '.join(flagged_codes)}"
                )
            if critic_report.get("missing_fields"):
                missing = ", ".join(critic_report["missing_fields"][:5])
                actions.append(f"Bổ sung dữ liệu: {missing}")

        if recommendations:
            top_choice = recommendations[0]
            actions.append(
                f"Xem xét sử dụng {top_choice['antibiotic_name']} "
                f"(P(sensitive)={top_choice['sensitive_probability']:.2f})"
            )
        else:
            actions.append("Chưa có khuyến nghị điều trị rõ ràng, cần hội chẩn.")

        return {
            "primary_actions": actions,
            "therapy_recommendations": recommendations,
        }




