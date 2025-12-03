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
    classification_report,
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
    """
    Core multi-output classifier for antibiotic resistance prediction.
    """

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
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        base_model = self._build_base_model(random_state)
        self.model = MultiOutputClassifier(base_model, n_jobs=-1)

        print(f"Đang huấn luyện mô hình {self.model_type}...")
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_proba = self.model.predict_proba(X_train)
        test_proba = self.model.predict_proba(X_test)

        train_exact_match = accuracy_score(y_train, train_pred)
        test_exact_match = accuracy_score(y_test, test_pred)

        train_hamming_loss_val = hamming_loss(y_train, train_pred)
        test_hamming_loss_val = hamming_loss(y_test, test_pred)
        train_hamming_acc = 1 - train_hamming_loss_val
        test_hamming_acc = 1 - test_hamming_loss_val

        train_per_label_acc = {}
        test_per_label_acc = {}
        for idx, col in enumerate(y.columns):
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
        for idx, col in enumerate(y.columns):
            try:
                train_auc_val = roc_auc_score(
                    y_train.iloc[:, idx],
                    train_proba[idx][:, 1],
                )
                test_auc_val = roc_auc_score(
                    y_test.iloc[:, idx],
                    test_proba[idx][:, 1],
                )
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
            "train_exact_match_accuracy": train_exact_match,
            "test_exact_match_accuracy": test_exact_match,
            "train_hamming_accuracy": train_hamming_acc,
            "test_hamming_accuracy": test_hamming_acc,
            "train_per_label_accuracy": train_per_label_acc,
            "test_per_label_accuracy": test_per_label_acc,
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
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_auc_mean": train_auc_mean,
            "test_auc_mean": test_auc_mean,
            "model_type": self.model_type,
        }

        return results

    def predict(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.is_trained:
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
        print(f"Mô hình đã được tải từ: {filepath}")


class AntibioticPredictionAgent:
    """
    Agent 2: wrap ResistancePredictorAgent with feature-column management.
    """

    def __init__(
        self,
        predictor: Optional[ResistancePredictorAgent] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        self.predictor = predictor or ResistancePredictorAgent()
        self.feature_columns = feature_columns

    @property
    def is_trained(self) -> bool:
        return bool(self.predictor and self.predictor.is_trained)

    def set_feature_columns(self, columns: List[str]):
        self.feature_columns = columns

    def load_model(self, model_path: str):
        self.predictor.load_model(model_path)

    def save_model(self, model_path: str):
        self.predictor.save_model(model_path)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict:
        results = self.predictor.train(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        self.feature_columns = X.columns.tolist()
        return results

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

