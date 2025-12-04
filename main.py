"""
MAS Clinical Decision System - sử dụng đúng 5 agent theo ý tưởng MAS
"""

from __future__ import annotations

import os
import joblib
import pandas as pd
from typing import Dict, Optional, Union

from agents.patient_data_agent import PatientDataAgent, RecordType
from agents.prediction_agent import (
    AntibioticPredictionAgent,
    ModelSelectionAgent,
)
from agents.explainability_agent import ExplainabilityEvaluationAgent
from agents.critic_agent import CriticAgent
from agents.decision_agent import DecisionAgent


class ClinicalDecisionPipeline:
    """
    Lightweight pipeline wiring the 5 MAS agents for inference.
    """

    def __init__(
        self,
        data_agent: Optional[PatientDataAgent] = None,
        prediction_agent: Optional[Union[AntibioticPredictionAgent, ModelSelectionAgent]] = None,
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


class MASClinicalDecisionSystem:
    """
    Orchestrator chính dựa trên 5 agent lõi:
    1. PatientDataAgent      → Nhập & tiền xử lý dữ liệu (Agent 1)
    2. AntibioticPrediction  → Huấn luyện & dự đoán (Agent 2)
    3. ExplainabilityAgent   → Đánh giá & giải thích (Agent 3)
    4. CriticAgent           → Phát hiện trường hợp thiếu chắc chắn
    5. DecisionAgent         → Gợi ý điều trị hoặc xét nghiệm bổ sung
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        model_dir: str = "models",
        auto_select_model: bool = False,
    ):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "mas_model.pkl")
        self.state_path = os.path.join(model_dir, "mas_state.joblib")
        self.auto_select_model = auto_select_model

        # Khởi tạo các agent cốt lõi
        if auto_select_model:
            # Sử dụng ModelSelectionAgent để tự động chọn mô hình tốt nhất
            self.prediction_agent = ModelSelectionAgent()
        else:
            # Sử dụng một agent cụ thể
            predictor_agent = AntibioticPredictionAgent(
                predictor=None,
            )
            if predictor_agent.predictor.model_type != model_type:
                predictor_agent.predictor.model_type = model_type
            self.prediction_agent = predictor_agent

        self.data_agent = PatientDataAgent()
        self.explain_agent = ExplainabilityEvaluationAgent()
        self.critic_agent = CriticAgent()
        self.decision_agent = DecisionAgent()

        # Pipeline inference dùng chung các agent
        self.pipeline = ClinicalDecisionPipeline(
            data_agent=self.data_agent,
            prediction_agent=self.prediction_agent,
            explain_agent=self.explain_agent,
            critic_agent=self.critic_agent,
            decision_agent=self.decision_agent,
        )

        self.feature_columns: Optional[list[str]] = None
        self.is_trained = False

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def train(
        self,
        csv_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        auto_select_model: bool = False,
        scoring_metric: str = "composite",
    ) -> Dict:
        """
        Huấn luyện toàn bộ MAS pipeline từ file CSV.
        
        Args:
            csv_path: Đường dẫn đến file CSV chứa dữ liệu huấn luyện
            test_size: Tỷ lệ test set (mặc định: 0.2)
            random_state: Random seed (mặc định: 42)
            auto_select_model: Nếu True, sẽ huấn luyện cả 3 mô hình và tự động chọn mô hình tốt nhất
            scoring_metric: Metric để chọn mô hình tốt nhất khi auto_select_model=True
                - "composite": Kết hợp test_accuracy, test_precision, test_recall, test_f1, test_auc_mean (mặc định)
                - "accuracy": Chỉ dùng test_accuracy
                - "precision": Chỉ dùng test_precision
                - "recall": Chỉ dùng test_recall
                - "f1": Chỉ dùng test_f1
                - "auc": Chỉ dùng test_auc_mean
        """
        print("=" * 80)
        print("BẮT ĐẦU HUẤN LUYỆN MAS CLINICAL DECISION SYSTEM")
        print("=" * 80)

        df = pd.read_csv(csv_path)
        X, y, feature_cols = self.data_agent.prepare_training_dataset(df)
        print(f"  ✓ Dữ liệu huấn luyện: {len(X)} mẫu, {len(feature_cols)} đặc trưng")

        y = y.fillna(0).astype(int)
        
        if auto_select_model:
            print("\n  ⚙️  Chế độ tự động chọn mô hình tốt nhất được kích hoạt")
            print("  → Sử dụng 3 agent riêng biệt: XGBoost, RandomForest, GradientBoosting")
            if not isinstance(self.prediction_agent, ModelSelectionAgent):
                # Nếu chưa khởi tạo ModelSelectionAgent, tạo mới
                self.prediction_agent = ModelSelectionAgent()
                self.pipeline.prediction_agent = self.prediction_agent
            
            results = self.prediction_agent.train_and_select_best(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                scoring_metric=scoring_metric,
            )
        else:
            if isinstance(self.prediction_agent, ModelSelectionAgent):
                # Nếu đang dùng ModelSelectionAgent nhưng không muốn auto_select, chuyển về single agent
                predictor_agent = AntibioticPredictionAgent(predictor=None)
                predictor_agent.predictor.model_type = "xgboost"
                self.prediction_agent = predictor_agent
                self.pipeline.prediction_agent = self.prediction_agent
            
            results = self.prediction_agent.train(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )

        self.feature_columns = feature_cols
        self.prediction_agent.set_feature_columns(feature_cols)
        self.pipeline.prediction_agent.set_feature_columns(feature_cols)

        self.is_trained = True
        self._ensure_model_dir()
        self.prediction_agent.save_model(self.model_path)
        self.save_state(self.state_path)

        print("\nHUẤN LUYỆN HOÀN TẤT. MÔ HÌNH ĐÃ ĐƯỢC LƯU!")
        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_state(self, filepath: Optional[str] = None):
        if not self.is_trained:
            raise ValueError("Không thể lưu trạng thái khi mô hình chưa được huấn luyện.")

        path = filepath or self.state_path
        self._ensure_model_dir()

        # Xác định model_type dựa trên loại agent
        if isinstance(self.prediction_agent, ModelSelectionAgent):
            model_type = self.prediction_agent.selected_model_type or "unknown"
        else:
            model_type = self.prediction_agent.predictor.model_type
        
        state = {
            "feature_columns": self.feature_columns,
            "data_agent": self.data_agent,
            "model_type": model_type,
            "auto_select_model": self.auto_select_model,
        }
        joblib.dump(state, path)
        print(f"  ✓ Đã lưu trạng thái MAS tại {path}")

    def load(
        self,
        model_path: Optional[str] = None,
        state_path: Optional[str] = None,
    ):
        """
        Load mô hình + trạng thái đã lưu.
        """
        model_file = model_path or self.model_path
        state_file = state_path or self.state_path

        if not os.path.exists(model_file) or not os.path.exists(state_file):
            raise FileNotFoundError("Không tìm thấy file mô hình hoặc state để load.")

        state = joblib.load(state_file)
        auto_select = state.get("auto_select_model", False)
        model_type = state.get("model_type", "xgboost")
        
        # Khởi tạo lại agent dựa trên state đã lưu
        if auto_select:
            if not isinstance(self.prediction_agent, ModelSelectionAgent):
                self.prediction_agent = ModelSelectionAgent()
                self.pipeline.prediction_agent = self.prediction_agent
            # Load model cho agent được chọn
            self.prediction_agent.load_model(model_file, model_type)
        else:
            if isinstance(self.prediction_agent, ModelSelectionAgent):
                predictor_agent = AntibioticPredictionAgent(predictor=None)
                predictor_agent.predictor.model_type = model_type
                self.prediction_agent = predictor_agent
                self.pipeline.prediction_agent = self.prediction_agent
            self.prediction_agent.load_model(model_file)
        
        self.feature_columns = state.get("feature_columns")
        self.auto_select_model = auto_select
        self.prediction_agent.set_feature_columns(self.feature_columns or [])
        self.pipeline.prediction_agent.set_feature_columns(self.feature_columns or [])

        if "data_agent" in state:
            self.data_agent = state["data_agent"]
            self.pipeline.data_agent = self.data_agent

        self.is_trained = True
        print("  ✓ Hệ thống MAS đã được khôi phục từ trạng thái lưu trữ.")

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------
    def predict(
        self,
        patient_record: Dict,
        bacteria_metadata: Optional[Dict] = None,
        antibiotic_panel: Optional[Dict] = None,
    ) -> Dict:
        if not self.is_trained:
            raise ValueError("Hệ thống chưa được huấn luyện. Vui lòng gọi train() hoặc load().")

        result = self.pipeline.run(
            patient_record=patient_record,
            bacteria_metadata=bacteria_metadata,
            antibiotic_panel=antibiotic_panel,
        )
        return result

    # ------------------------------------------------------------------
    def _ensure_model_dir(self):
        os.makedirs(self.model_dir, exist_ok=True)


def main():
    """
    Ví dụ chạy end-to-end: train + predict 1 bệnh nhân mẫu.
    """
    # Sử dụng auto_select_model=True để tự động chọn mô hình tốt nhất từ 3 agent
    system = MASClinicalDecisionSystem(
        model_type="xgboost",
        auto_select_model=True  # Sử dụng 3 agent riêng biệt
    )

    csv_path = "data/Bacteria_dataset_Multiresictance.csv"
    if not os.path.exists(csv_path):
        csv_path = "Bacteria_dataset_Multiresictance.csv"

    if not os.path.exists(csv_path):
        print("❌ Không tìm thấy file dữ liệu huấn luyện.")
        return

    system.train(
        csv_path, 
        test_size=0.2, 
        random_state=42,
        auto_select_model=True,  # Đã được set trong __init__
        scoring_metric="composite"  # Hoặc "accuracy", "f1", "auc"
    )

    sample_patient = {
        "age/gender": "45/F",
        "Souches": "S999 Escherichia coli",
        "Diabetes": "Yes",
        "Hypertension": "No",
        "Hospital_before": "Yes",
        "Infection_Freq": 2.0,
        "Collection_Date": "2024-01-15",    
    }

    print("\n" + "=" * 80)
    print("DỰ ĐOÁN CHO BỆNH NHÂN MẪU")
    print("=" * 80)

    result = system.predict(sample_patient)
    print("Predictions:", result["predictions"])
    print("Mean probability:", round(pd.Series(result["probabilities"]).mean(), 3))
    print("\n--- ACTION PLAN ---")
    for action in result["decision"]["primary_actions"]:
        print(" •", action)


if __name__ == "__main__":
    main()
