"""
Prediction agent module - model training & inference for MAS
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb


class ResistancePredictorAgent:

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.antibiotic_columns = [
            "AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO", "IPM", "GEN", "AN",
            "Acide nalidixique", "ofx", "CIP", "C", "Co-trimoxazole",
            "Furanes", "colistine",
        ]
        self.is_trained = False

    def _build_base_model(self, random_state: int):
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1,
            )
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            )
        # Sử dụng GradientBoostingClassifier cho các trường hợp còn lại
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
        )

    def train_with_splits(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        random_state: int = 42,
    ) -> Dict:
        """
        Huấn luyện với train/test set đã được chia sẵn và tính toán metrics.
        """
        base_model = self._build_base_model(random_state)
        self.model = MultiOutputClassifier(base_model, n_jobs=-1)

        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_proba = self.model.predict_proba(X_train)
        test_proba = self.model.predict_proba(X_test)

        # Tính toán tất cả các metrics... (Logic giữ nguyên)
        train_exact_match = accuracy_score(y_train, train_pred)
        test_exact_match = accuracy_score(y_test, test_pred)
        train_hamming_loss_val = hamming_loss(y_train, train_pred)
        test_hamming_loss_val = hamming_loss(y_test, test_pred)
        train_hamming_acc = 1 - train_hamming_loss_val
        test_hamming_acc = 1 - test_hamming_loss_val
        
        train_per_label_acc = {}
        test_per_label_acc = {}
        for idx, col in enumerate(y_train.columns):
            train_per_label_acc[col] = accuracy_score(y_train.iloc[:, idx], train_pred[:, idx])
            test_per_label_acc[col] = accuracy_score(y_test.iloc[:, idx], test_pred[:, idx])
            
        train_avg_per_label_acc = np.mean(list(train_per_label_acc.values()))
        test_avg_per_label_acc = np.mean(list(test_per_label_acc.values()))
        train_jaccard = jaccard_score(y_train, train_pred, average="macro", zero_division=0)
        test_jaccard = jaccard_score(y_test, test_pred, average="macro", zero_division=0)
        train_precision = precision_score(y_train, train_pred, average="macro", zero_division=0)
        test_precision = precision_score(y_test, test_pred, average="macro", zero_division=0)
        train_recall = recall_score(y_train, train_pred, average="macro", zero_division=0)
        test_recall = recall_score(y_test, test_pred, average="macro", zero_division=0)
        train_f1 = f1_score(y_train, train_pred, average="macro", zero_division=0)
        test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
        
        train_auc = {}
        test_auc = {}
        train_auc_mean = 0
        test_auc_mean = 0
        auc_count = 0
        for idx, col in enumerate(y_train.columns):
            try:
                train_auc_val = roc_auc_score(y_train.iloc[:, idx], train_proba[idx][:, 1])
                test_auc_val = roc_auc_score(y_test.iloc[:, idx], test_proba[idx][:, 1])
                train_auc[col] = train_auc_val
                test_auc[col] = test_auc_val
                train_auc_mean += train_auc_val
                test_auc_mean += test_auc_val
                auc_count += 1
            except Exception:
                train_auc[col] = None
                test_auc[col] = None
        if auc_count > 0:
            train_auc_mean /= auc_count
            test_auc_mean /= auc_count

        self.is_trained = True

        results = {
            "train_accuracy": train_exact_match,
            "test_accuracy": test_exact_match,
            "train_hamming_accuracy": train_hamming_acc,
            "test_hamming_accuracy": test_hamming_acc,
            "train_avg_per_label_accuracy": train_avg_per_label_acc,
            "test_avg_per_label_accuracy": test_avg_per_label_acc,
            "train_jaccard_score": train_jaccard,
            "test_jaccard_score": test_jaccard,
            "train_precision": train_precision,
            "test_precision": test_precision,
            "train_recall": train_recall,
            "test_recall": test_recall,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "train_auc_mean": train_auc_mean,
            "test_auc_mean": test_auc_mean,
            "model_type": self.model_type,
        }

        return results

    def predict(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.is_trained:
            # Sẽ được ModelSelectionAgent kiểm tra trước
            raise ValueError("Mô hình chưa được huấn luyện!")

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        pred_df = pd.DataFrame(
            predictions,
            columns=self.antibiotic_columns,
            index=X.index,
        )

        proba_dict = {}
        for idx, col in enumerate(self.antibiotic_columns):
            if idx < len(probabilities):
                # Lấy xác suất của nhãn 'positive' (nhãn 1)
                proba_dict[col] = probabilities[idx][:, 1]

        proba_df = pd.DataFrame(proba_dict, index=X.index) 
        return pred_df, proba_df

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        _, proba_df = self.predict(X)
        return proba_df

    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("Không có mô hình để lưu!")
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Mô hình đã được lưu tại: {filepath}")

    def load_model(self, filepath: str):
        self.model = joblib.load(filepath)
        self.is_trained = True
        # Ghi đè self.model_type bằng loại mô hình được tải nếu cần (không cần thiết vì ModelSelectionAgent quản lý)
        print(f"Mô hình đã được tải từ: {filepath}")


class AntibioticPredictionAgent:
    """
    Agent 2: wrap ResistancePredictorAgent with feature-column management.
    Chỉ giữ lại chức năng quản lý feature columns và predict.
    """

    def __init__(
        self,
        predictor: Optional[ResistancePredictorAgent] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        # Đảm bảo có predictor, nhưng không cần mặc định tạo mới
        self.predictor = predictor
        self.feature_columns = feature_columns

    @property
    def is_trained(self) -> bool:
        # Kiểm tra trạng thái huấn luyện của predictor
        return bool(self.predictor and self.predictor.is_trained)

    def set_feature_columns(self, columns: List[str]):
        self.feature_columns = columns
        
    def predict(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.predictor or not self.predictor.is_trained:
            # ModelSelectionAgent sẽ đảm bảo điều này
            raise ValueError("Mô hình chưa được huấn luyện hoặc load.")

        prepared = feature_df.copy()
        if self.feature_columns:
            # Logic đảm bảo feature columns nhất quán
            for col in self.feature_columns:
                if col not in prepared.columns:
                    prepared[col] = 0
            prepared = prepared[self.feature_columns]

        # Đảm bảo tất cả các cột là dạng số và xử lý NaN
        prepared = prepared.apply(pd.to_numeric, errors="coerce").fillna(0)
        predictions, probabilities = self.predictor.predict(prepared)
        return predictions, probabilities


# ============================================================================
# 3 Agent riêng biệt cho từng mô hình 
# ============================================================================

class XGBoostPredictionAgent(AntibioticPredictionAgent):
    """ Agent chuyên dụng cho mô hình XGBoost. """
    
    def __init__(self, feature_columns: Optional[List[str]] = None):
        predictor = ResistancePredictorAgent(model_type="xgboost")
        super().__init__(predictor=predictor, feature_columns=feature_columns)
        self.agent_name = "XGBoost Agent"


class RandomForestPredictionAgent(AntibioticPredictionAgent):
    """ Agent chuyên dụng cho mô hình Random Forest. """
    
    def __init__(self, feature_columns: Optional[List[str]] = None):
        predictor = ResistancePredictorAgent(model_type="random_forest")
        super().__init__(predictor=predictor, feature_columns=feature_columns)
        self.agent_name = "Random Forest Agent"


class GradientBoostingPredictionAgent(AntibioticPredictionAgent):
    """ Agent chuyên dụng cho mô hình Gradient Boosting. """
    
    def __init__(self, feature_columns: Optional[List[str]] = None):
        predictor = ResistancePredictorAgent(model_type="gradient_boosting")
        super().__init__(predictor=predictor, feature_columns=feature_columns)
        self.agent_name = "Gradient Boosting Agent"


# ============================================================================
# Orchestrator Agent để điều phối và TỰ ĐỘNG chọn mô hình tốt nhất
# ============================================================================

class ModelSelectionAgent:
    """
    Agent điều phối chỉ cho phép huấn luyện và tự động chọn mô hình tốt nhất.
    """
    
    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns
        
        # Khởi tạo 3 agent riêng biệt
        self.xgboost_agent = XGBoostPredictionAgent(feature_columns=feature_columns)
        self.random_forest_agent = RandomForestPredictionAgent(feature_columns=feature_columns)
        self.gradient_boosting_agent = GradientBoostingPredictionAgent(feature_columns=feature_columns)
        
        # Agent được chọn
        self.selected_agent: Optional[AntibioticPredictionAgent] = None
        self.selected_model_type: Optional[str] = None
        
    @property
    def is_trained(self) -> bool:
        return self.selected_agent is not None and self.selected_agent.is_trained
    
    def set_feature_columns(self, columns: List[str]):
        """Cập nhật feature columns cho tất cả các agent."""
        self.feature_columns = columns
        self.xgboost_agent.set_feature_columns(columns)
        self.random_forest_agent.set_feature_columns(columns)
        self.gradient_boosting_agent.set_feature_columns(columns)
    
    def train_and_select_best(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        scoring_metric: str = "composite",
    ) -> Dict:
        """
        Huấn luyện cả 3 agent và tự động chọn agent tốt nhất.
        """
        print("=" * 80)
        print("HUẤN LUYỆN VÀ TỰ ĐỘNG CHỌN MÔ HÌNH TỐT NHẤT")
        print("=" * 80)
        
        # Chia dữ liệu một lần để đảm bảo công bằng
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Cập nhật feature columns
        self.set_feature_columns(X.columns.tolist())
        
        agents = {
            "xgboost": self.xgboost_agent,
            "random_forest": self.random_forest_agent,
            "gradient_boosting": self.gradient_boosting_agent,
        }
        
        all_results = {}
        
        # Huấn luyện từng agent
        for model_type, agent in agents.items():
            print(f"\n{'='*60}")
            print(f"Đang huấn luyện: {agent.agent_name}")
            print(f"{'='*60}")
            
            # Gọi phương thức train_with_splits 
            results = agent.predictor.train_with_splits( 
                X_train, X_test, y_train, y_test, random_state=random_state
            )
            
            all_results[model_type] = {
                "results": results,
                "agent": agent,
            }
            
            print(f"  ✓ Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"  ✓ Test F1 Score: {results['test_f1']:.4f}")
            print(f"  ✓ Test AUC Mean: {results['test_auc_mean']:.4f}")
        
        # Tính điểm và chọn agent tốt nhất (Logic giữ nguyên)
        scores = {}
        for model_type, data in all_results.items():
            results = data["results"]
            
            if scoring_metric == "accuracy":
                score = results["test_accuracy"]
            elif scoring_metric == "precision":
                score = results["test_precision"]
            elif scoring_metric == "recall":
                score = results["test_recall"]
            elif scoring_metric == "f1":
                score = results["test_f1"]
            elif scoring_metric == "auc":
                score = results["test_auc_mean"]
            else:  # composite (mặc định)
                # Kết hợp: 25% accuracy, 20% precision, 20% recall, 20% f1, 15% auc
                score = (
                    0.25 * results["test_accuracy"] +
                    0.20 * results["test_precision"] +
                    0.20 * results["test_recall"] +
                    0.20 * results["test_f1"] +
                    0.15 * results["test_auc_mean"]
                )
            
            scores[model_type] = score
        
        best_model_type = max(scores, key=scores.get)
        best_agent = all_results[best_model_type]["agent"]
        best_results = all_results[best_model_type]["results"]
        
        # Cập nhật agent được chọn
        self.selected_agent = best_agent
        self.selected_model_type = best_model_type
        
        print("\n" + "=" * 100)
        print("KẾT QUẢ SO SÁNH CÁC AGENT")
        print("=" * 100)
        print(f"{'Agent':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10} {'Score':<10}")
        print("-" * 100)
        for model_type, data in all_results.items():
            agent = data["agent"]
            r = data["results"]
            score = scores[model_type]
            marker = " ← ĐƯỢC CHỌN" if model_type == best_model_type else ""
            print(
                f"{agent.agent_name:<30} "
                f"{r['test_accuracy']:<10.4f} "
                f"{r['test_precision']:<10.4f} "
                f"{r['test_recall']:<10.4f} "
                f"{r['test_f1']:<10.4f} "
                f"{r['test_auc_mean']:<10.4f} "
                f"{score:<10.4f}{marker}"
            )
        print("=" * 100)
        print(f"\n✓ Đã chọn agent: {best_agent.agent_name}")  
        
        best_results["model_comparison"] = {
             "all_results": {
                mt: {
                    "agent_name": agents[mt].agent_name,
                    "test_accuracy": all_results[mt]["results"]["test_accuracy"],
                    "test_precision": all_results[mt]["results"]["test_precision"],
                    "test_recall": all_results[mt]["results"]["test_recall"],
                    "test_f1": all_results[mt]["results"]["test_f1"],
                    "test_auc_mean": all_results[mt]["results"]["test_auc_mean"],
                    "score": scores[mt],
                }
                for mt in agents.keys()
            },
            "best_model": best_model_type,
            "best_agent_name": best_agent.agent_name,
            "scoring_metric": scoring_metric,
        }
        
        return best_results
    
    def predict(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sử dụng agent được chọn để dự đoán."""
        if not self.is_trained:
            raise ValueError("Chưa có agent nào được chọn. Vui lòng gọi train_and_select_best() trước.")
        
        return self.selected_agent.predict(feature_df)
    
    def save_model(self, model_path: str):
        """Lưu mô hình của agent được chọn."""
        if not self.is_trained:
            raise ValueError("Không có mô hình để lưu!")
        self.selected_agent.predictor.save_model(model_path)
    
    def load_model(self, model_path: str, model_type: str, feature_columns: List[str]):
        """
        Load mô hình và chọn làm agent chính. (Vẫn giữ lại load để triển khai mô hình đã chọn)
        
        Args:
            model_path: Đường dẫn đến file mô hình
            model_type: Loại mô hình ("xgboost", "random_forest", "gradient_boosting")
            feature_columns: Danh sách các cột feature dùng khi huấn luyện mô hình này
        """
        agents = {
            "xgboost": self.xgboost_agent,
            "random_forest": self.random_forest_agent,
            "gradient_boosting": self.gradient_boosting_agent,
        }
        
        if model_type not in agents:
            raise ValueError(f"Model type không hợp lệ: {model_type}")
        
        agent = agents[model_type]
        agent.predictor.load_model(model_path) # Load vào predictor
        agent.set_feature_columns(feature_columns) # Cài đặt lại feature columns
        self.set_feature_columns(feature_columns)
        self.selected_agent = agent
        self.selected_model_type = model_type
        print(f"✓ Đã load mô hình {model_type} và chọn làm agent chính")