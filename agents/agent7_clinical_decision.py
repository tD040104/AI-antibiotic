"""
Agent Suite - Clinical Decision Support Extension

Bao gồm các agent mới:
1. PatientDataAgent: Nhập và tiền xử lý dữ liệu bệnh nhân + vi khuẩn + kháng sinh
2. AntibioticPredictionAgent: Bao bọc mô hình dự đoán kháng/nhạy
3. ExplainabilityEvaluationAgent: Đánh giá + giải thích (SHAP/LIME)
4. CriticAgent: Phát hiện trường hợp thiếu chắc chắn hoặc thiếu dữ liệu
5. DecisionAgent: Gợi ý điều trị hoặc yêu cầu thêm xét nghiệm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from agents.agent1_data_cleaner import DataCleanerAgent
from agents.agent2_feature_engineer import FeatureEngineerAgent
from agents.agent3_resistance_predictor import ResistancePredictorAgent
from agents.agent4_treatment_recommender import TreatmentRecommenderAgent
from agents.agent5_explainability import ExplainabilityAgent, SHAP_AVAILABLE


RecordType = Union[Dict, List[Dict], pd.Series, pd.DataFrame]


class PatientDataAgent:
    """
    Agent 1 (mới): Nhập và tiền xử lý bệnh nhân + metadata vi khuẩn + panel kháng sinh.
    """

    def __init__(
        self,
        data_cleaner: Optional[DataCleanerAgent] = None,
        feature_engineer: Optional[FeatureEngineerAgent] = None,
    ):
        self.data_cleaner = data_cleaner or DataCleanerAgent()
        self.feature_engineer = feature_engineer or FeatureEngineerAgent()

    def ingest(
        self,
        patient_records: RecordType,
        bacteria_metadata: Optional[RecordType] = None,
        antibiotic_panel: Optional[RecordType] = None,
    ) -> pd.DataFrame:
        """
        Chuẩn hóa tất cả nguồn dữ liệu thành một DataFrame duy nhất.
        """
        patient_df = self._ensure_dataframe(patient_records, name="patient_records")

        if bacteria_metadata is not None:
            bacteria_df = self._ensure_dataframe(bacteria_metadata, name="bacteria_metadata")
            patient_df = self._merge_metadata(patient_df, bacteria_df, prefix="Bacteria_")

        if antibiotic_panel is not None:
            antibiotics_df = self._ensure_dataframe(antibiotic_panel, name="antibiotic_panel")
            patient_df = self._merge_metadata(patient_df, antibiotics_df, prefix="LabPanel_")

        return patient_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tiền xử lý (làm sạch + feature engineering) để sẵn sàng cho mô hình.
        """
        cleaned = self.data_cleaner.clean(df)
        features = self.feature_engineer.engineer_features(cleaned)
        return features

    def prepare(
        self,
        patient_records: RecordType,
        bacteria_metadata: Optional[RecordType] = None,
        antibiotic_panel: Optional[RecordType] = None,
    ) -> pd.DataFrame:
        """
        Pipeline đầy đủ: ingest -> preprocess.
        """
        merged = self.ingest(patient_records, bacteria_metadata, antibiotic_panel)
        processed = self.preprocess(merged)
        return processed

    @staticmethod
    def _ensure_dataframe(records: RecordType, name: str) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records.copy()
        if isinstance(records, pd.Series):
            return pd.DataFrame([records.to_dict()])
        if isinstance(records, dict):
            return pd.DataFrame([records])
        if isinstance(records, list):
            return pd.DataFrame(records)
        raise ValueError(f"{name} phải là dict/list/pd.Series/pd.DataFrame")

    @staticmethod
    def _merge_metadata(
        patient_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        if metadata_df.empty:
            return patient_df

        merged = patient_df.copy()
        metadata_row = metadata_df.iloc[0].to_dict()
        for col, value in metadata_row.items():
            new_col = col if col in merged.columns else f"{prefix}{col}"
            if new_col not in merged.columns or merged[new_col].isna().all():
                merged[new_col] = value
        return merged


class AntibioticPredictionAgent:
    """
    Agent 2 (mới): Bao bọc mô hình dự đoán kháng/nhạy.
    """

    def __init__(
        self,
        predictor: Optional[ResistancePredictorAgent] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        self.predictor = predictor or ResistancePredictorAgent()
        self.feature_columns = feature_columns

    def set_feature_columns(self, columns: List[str]):
        self.feature_columns = columns

    def load_model(self, model_path: str):
        self.predictor.load_model(model_path)

    def predict(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.predictor.is_trained:
            raise ValueError("Mô hình chưa được huấn luyện hoặc load.")

        prepared = feature_df.copy()
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in prepared.columns:
                    prepared[col] = 0
            prepared = prepared[self.feature_columns]

        prepared = prepared.apply(pd.to_numeric, errors="coerce").fillna(0)
        predictions, probabilities = self.predictor.predict(prepared)
        return predictions, probabilities


class ExplainabilityEvaluationAgent:
    """
    Agent 3 (mới): Đánh giá + giải thích dự đoán (SHAP/LIME).
    """

    def __init__(
        self,
        explainability_agent: Optional[ExplainabilityAgent] = None,
    ):
        self.explainability_agent = explainability_agent or ExplainabilityAgent()

    def explain(
        self,
        patient_features: pd.Series,
        predictions: pd.DataFrame,
        probabilities: pd.DataFrame,
        feature_importance: Optional[pd.DataFrame] = None,
        shap_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        return self.explainability_agent.explain_prediction(
            patient_features,
            predictions,
            probabilities,
            feature_importance=feature_importance,
            shap_values=shap_values if SHAP_AVAILABLE else None,
            feature_names=feature_names,
        )

    def evaluate_confidence(
        self,
        probabilities: pd.DataFrame,
        ground_truth: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Đánh giá nhanh độ tin cậy và hiệu suất.
        """
        stats = {}
        if probabilities.empty:
            return {"mean_confidence": 0.0, "uncertain_fraction": 1.0}

        proba_row = probabilities.iloc[0]
        stats["mean_confidence"] = float(proba_row.mean())
        stats["median_confidence"] = float(proba_row.median())

        uncertain_mask = (proba_row >= 0.4) & (proba_row <= 0.6)
        stats["uncertain_fraction"] = float(uncertain_mask.sum() / len(proba_row))

        if ground_truth is not None and not ground_truth.empty:
            gt_row = ground_truth.iloc[0]
            pred_labels = (proba_row >= 0.5).astype(int)
            stats["observed_accuracy"] = float((pred_labels == gt_row).mean())

        return stats


@dataclass
class CriticFlag:
    antibiotic: str
    probability: float
    reason: str


class CriticAgent:
    """
    Critic Agent: Phát hiện trường hợp không chắc chắn hoặc dữ liệu bất thường.
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
            elif proba <= 0.2 or proba >= 0.8:
                continue  # confident cases

        missing = missing_required_fields or []
        if patient_features is not None:
            for col, value in patient_features.items():
                if pd.isna(value) or value in ["", "Unknown"]:
                    missing.append(col)

        return {
            "flags": flags,
            "missing_fields": sorted(set(missing)),
            "needs_additional_tests": len(flags) > 0 or len(missing) > 0,
        }


class DecisionAgent:
    """
    Decision Agent: Kết hợp Critic + Treatment recommender để đưa ra hành động.
    """

    def __init__(
        self,
        treatment_agent: Optional[TreatmentRecommenderAgent] = None,
    ):
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


class ClinicalDecisionPipeline:
    """
    Pipeline end-to-end sử dụng các agent mới.
    """

    def __init__(
        self,
        data_agent: Optional[PatientDataAgent] = None,
        prediction_agent: Optional[AntibioticPredictionAgent] = None,
        explain_agent: Optional[ExplainabilityEvaluationAgent] = None,
        critic_agent: Optional[CriticAgent] = None,
        decision_agent: Optional[DecisionAgent] = None,
    ):
        self.data_agent = data_agent or PatientDataAgent()
        self.prediction_agent = prediction_agent or AntibioticPredictionAgent()
        self.explain_agent = explain_agent or ExplainabilityEvaluationAgent()
        self.critic_agent = critic_agent or CriticAgent()
        self.decision_agent = decision_agent or DecisionAgent()

    def run(
        self,
        patient_record: RecordType,
        bacteria_metadata: Optional[RecordType] = None,
        antibiotic_panel: Optional[RecordType] = None,
    ) -> Dict:
        features_df = self.data_agent.prepare(
            patient_record,
            bacteria_metadata=bacteria_metadata,
            antibiotic_panel=antibiotic_panel,
        )

        predictions, probabilities = self.prediction_agent.predict(features_df)
        patient_series = features_df.iloc[0]

        critic_report = self.critic_agent.review(probabilities, patient_series)
        explanation = self.explain_agent.explain(
            patient_series,
            predictions,
            probabilities,
        )
        decision = self.decision_agent.decide(
            probabilities,
            critic_report,
            patient_series,
        )

        return {
            "features": patient_series.to_dict(),
            "predictions": predictions.iloc[0].to_dict(),
            "probabilities": probabilities.iloc[0].to_dict(),
            "explanation": explanation,
            "critic_report": critic_report,
            "decision": decision,
        }






