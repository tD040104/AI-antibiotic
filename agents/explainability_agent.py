"""
Explainability agent - SHAP/LIME style explanations + reports
Enhanced with Embedding and Transformer for decision making
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None  # type: ignore


class FeatureEmbedder:
    """Embedding layer for patient features using transformer architecture."""
    
    def __init__(self, input_dim: int = 50, embedding_dim: int = 128, hidden_dim: int = 256):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for FeatureEmbedder. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Simple transformer-like architecture
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Decision head
        self.decision_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def fit(self, X: np.ndarray):
        """Fit scaler on training data."""
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: np.ndarray) -> torch.Tensor:
        """Transform features to embeddings."""
        if not self.is_fitted:
            X_scaled = X
        else:
            X_scaled = self.scaler.transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled)
        # Add sequence dimension for transformer
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)  # [batch, 1, features]
        
        # Embedding
        embedded = self.embedding_layer(X_tensor.squeeze(1))
        embedded = embedded.unsqueeze(1)  # [batch, 1, embedding_dim]
        
        # Transformer encoding
        encoded = self.transformer(embedded)
        return encoded.squeeze(1)  # [batch, embedding_dim]
    
    def predict_decision(self, X: np.ndarray) -> float:
        """Predict decision score using transformer."""
        with torch.no_grad():
            encoded = self.transform(X)
            decision_score = self.decision_head(encoded)
            return decision_score.item()


class ExplainabilityAgent:
    """Base explainability + report generator with embedding and transformer."""

    def __init__(self, use_embedding: bool = True, embedding_dim: int = 128):
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
        
        self.use_embedding = use_embedding and TORCH_AVAILABLE
        self.embedder = None
        self.text_embedder = None
        
        if self.use_embedding:
            try:
                # Initialize feature embedder
                self.embedder = FeatureEmbedder(input_dim=50, embedding_dim=embedding_dim)
            except Exception:
                self.embedder = None
                self.use_embedding = False
            
            # Initialize text embedder if available
            if SENTENCE_TRANSFORMER_AVAILABLE:
                try:
                    self.text_embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                except:
                    self.text_embedder = None

    def _create_feature_embedding(self, patient_data: pd.Series, probabilities: pd.Series) -> np.ndarray:
        """Create embedding vector from patient features and probabilities."""
        # Combine patient features and probabilities
        feature_values = []
        
        # Patient features
        numeric_features = ['Age', 'Diabetes', 'Hypertension', 'Hospital_before', 
                           'Infection_Freq', 'Total_risk_factors']
        for feat in numeric_features:
            val = patient_data.get(feat, 0)
            feature_values.append(float(val) if not pd.isna(val) else 0.0)
        
        # Probabilities
        for ab in probabilities.index:
            feature_values.append(float(probabilities[ab]))
        
        # Pad or truncate to fixed size
        target_size = 50
        if len(feature_values) < target_size:
            feature_values.extend([0.0] * (target_size - len(feature_values)))
        else:
            feature_values = feature_values[:target_size]
        
        return np.array(feature_values).reshape(1, -1)
    
    def _generate_decision_from_embedding(
        self, 
        patient_data: pd.Series, 
        predictions: pd.Series, 
        probabilities: pd.Series
    ) -> Dict:
        """Generate decision using transformer-based embedding."""
        if not self.use_embedding or self.embedder is None:
            # Fallback to rule-based
            return self._generate_rule_based_decision(patient_data, predictions, probabilities)
        
        try:
            # Create feature embedding
            feature_vector = self._create_feature_embedding(patient_data, probabilities)
            
            # Get decision score from transformer
            decision_score = self.embedder.predict_decision(feature_vector)
            
            # Generate decision based on score
            if decision_score >= 0.7:
                decision_type = "high_confidence_treatment"
                recommended_ab = probabilities.idxmax()
            elif decision_score >= 0.4:
                decision_type = "moderate_confidence_treatment"
                recommended_ab = probabilities.idxmax()
            else:
                decision_type = "requires_additional_testing"
                recommended_ab = None
            
            return {
                "decision_type": decision_type,
                "decision_score": float(decision_score),
                "recommended_antibiotic": recommended_ab,
                "confidence": "high" if decision_score >= 0.7 else "moderate" if decision_score >= 0.4 else "low",
                "reasoning": f"Transformer-based decision score: {decision_score:.3f}"
            }
        except Exception as e:
            # Fallback on error
            return self._generate_rule_based_decision(patient_data, predictions, probabilities)
    
    def _generate_rule_based_decision(
        self, 
        patient_data: pd.Series, 
        predictions: pd.Series, 
        probabilities: pd.Series
    ) -> Dict:
        """Fallback rule-based decision."""
        max_prob = probabilities.max()
        recommended_ab = probabilities.idxmax()
        
        if max_prob >= 0.7:
            decision_type = "high_confidence_treatment"
            confidence = "high"
        elif max_prob >= 0.4:
            decision_type = "moderate_confidence_treatment"
            confidence = "moderate"
        else:
            decision_type = "requires_additional_testing"
            confidence = "low"
        
        return {
            "decision_type": decision_type,
            "decision_score": float(max_prob),
            "recommended_antibiotic": recommended_ab,
            "confidence": confidence,
            "reasoning": "Rule-based fallback decision"
        }

    def explain_prediction(
        self,
        patient_data: pd.Series,
        predictions: pd.DataFrame,
        probabilities: pd.DataFrame,
        feature_importance: Optional[pd.DataFrame] = None,
        shap_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        # Generate decision using embedding/transformer
        decision = self._generate_decision_from_embedding(
            patient_data, predictions.iloc[0], probabilities.iloc[0]
        )
        
        explanation = {
            "patient_summary": self._extract_patient_info(patient_data),
            "resistance_predictions": self._summarize_resistance(
                predictions.iloc[0], probabilities.iloc[0]
            ),
            "key_factors": self._identify_key_factors(
                patient_data, feature_importance, shap_values, feature_names
            ),
            "report": self._generate_natural_language_report(
                patient_data, predictions.iloc[0], probabilities.iloc[0]
            ),
            "decision": decision,  # Add decision output
        }
        return explanation

    def _extract_patient_info(self, patient_data: pd.Series) -> Dict:
        return {
            "age": patient_data.get("Age", "Unknown"),
            "gender": patient_data.get("Gender", "Unknown"),
            "bacteria": patient_data.get("Bacteria", "Unknown"),
            "diabetes": "Có" if patient_data.get("Diabetes", 0) == 1 else "Không",
            "hypertension": "Có" if patient_data.get("Hypertension", 0) == 1 else "Không",
            "hospital_before": "Có" if patient_data.get("Hospital_before", 0) == 1 else "Không",
            "infection_freq": patient_data.get("Infection_Freq", 0),
        }

    def _summarize_resistance(
        self,
        predictions: pd.Series,
        probabilities: pd.Series,
    ) -> Dict:
        resistant = []
        sensitive = []
        for antibiotic in predictions.index:
            pred = predictions[antibiotic]
            proba = probabilities[antibiotic]
            if pred == 1:
                sensitive.append({
                    "antibiotic": self.antibiotic_names.get(antibiotic, antibiotic),
                    "code": antibiotic,
                    "probability": round(proba, 3),
                })
            else:
                resistant.append({
                    "antibiotic": self.antibiotic_names.get(antibiotic, antibiotic),
                    "code": antibiotic,
                    "resistance_probability": round(1 - proba, 3),
                })
        return {
            "sensitive": sensitive,
            "resistant": resistant,
            "sensitive_count": len(sensitive),
            "resistant_count": len(resistant),
        }

    def _identify_key_factors(
        self,
        patient_data: pd.Series,
        feature_importance: Optional[pd.DataFrame] = None,
        shap_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict]:
        factors: List[Dict] = []

        if SHAP_AVAILABLE and shap_values is not None and feature_names is not None:
            if shap_values.ndim > 2:
                shap_values_sample = shap_values[:, 0, :]
                shap_sum = np.abs(shap_values_sample).mean(axis=0)
            else:
                shap_sum = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(shap_sum)[::-1][:10]
            for idx in top_indices:
                if idx < len(feature_names):
                    factors.append({
                        "feature": feature_names[idx],
                        "importance": float(shap_sum[idx]),
                        "type": "shap",
                    })
        elif feature_importance is not None and not feature_importance.empty:
            top_features = feature_importance.head(10)
            for _, row in top_features.iterrows():
                factors.append({
                    "feature": row["feature"],
                    "importance": float(row["importance"]),
                    "type": "model_importance",
                })

        clinical_factors = {
            "Diabetes": patient_data.get("Diabetes", 0),
            "Hypertension": patient_data.get("Hypertension", 0),
            "Hospital_before": patient_data.get("Hospital_before", 0),
            "Infection_Freq": patient_data.get("Infection_Freq", 0),
            "Total_risk_factors": patient_data.get("Total_risk_factors", 0),
            "Age": patient_data.get("Age", None),
        }
        for factor, value in clinical_factors.items():
            if value and value != 0:
                factors.append({
                    "feature": factor,
                    "value": value,
                    "type": "clinical",
                })

        return factors[:15]

    def _generate_natural_language_report(
        self,
        patient_data: pd.Series,
        predictions: pd.Series,
        probabilities: pd.Series,
    ) -> str:
        age = patient_data.get("Age", "Unknown")
        gender = patient_data.get("Gender", "Unknown")
        gender_vn = {"Female": "nữ", "Male": "nam"}.get(gender, "không xác định")
        bacteria = patient_data.get("Bacteria", "Unknown")

        risk_factors = []
        if patient_data.get("Diabetes", 0) == 1:
            risk_factors.append("tiểu đường")
        if patient_data.get("Hypertension", 0) == 1:
            risk_factors.append("tăng huyết áp")
        if patient_data.get("Hospital_before", 0) == 1:
            risk_factors.append("tiền sử nhập viện")
        risk_text = ", ".join(risk_factors) if risk_factors else "không có yếu tố nguy cơ đáng kể"

        sensitive_antibiotics = []
        resistant_antibiotics = []
        for antibiotic in predictions.index:
            if predictions[antibiotic] == 1:
                sensitive_antibiotics.append(self.antibiotic_names.get(antibiotic, antibiotic))
            else:
                if probabilities[antibiotic] < 0.3:
                    resistant_antibiotics.append(self.antibiotic_names.get(antibiotic, antibiotic))

        report = f"Bệnh nhân {gender_vn}, {age} tuổi, nhiễm {bacteria}"
        if risk_factors:
            report += f", có {risk_text}"
        report += ".\n\n"

        if resistant_antibiotics:
            report += f"Dự đoán kháng với: {', '.join(resistant_antibiotics[:5])}.\n"
        if sensitive_antibiotics:
            report += f"Dự đoán nhạy với: {', '.join(sensitive_antibiotics[:5])}.\n"

        if sensitive_antibiotics:
            first_sensitive = next((ab for ab in predictions.index if predictions[ab] == 1), None)
            if first_sensitive:
                proba_value = probabilities.get(first_sensitive, 0.0)
                report += (
                    f"\nKhuyến nghị: Xem xét sử dụng {self.antibiotic_names.get(first_sensitive, first_sensitive)} "
                    f"(xác suất nhạy: {proba_value:.1%})."
                )
        else:
            report += "\nCảnh báo: Tất cả các kháng sinh được kiểm tra đều có khả năng kháng cao."

        return report

    def generate_detailed_report(self, explanation: Dict) -> str:
        report = "=" * 80 + "\n"
        report += "BÁO CÁO PHÂN TÍCH KHÁNG KHÁNG SINH\n"
        report += "=" * 80 + "\n\n"

        info = explanation["patient_summary"]
        report += "THÔNG TIN BỆNH NHÂN:\n"
        report += f"Tuổi: {info['age']}\n"
        report += f"Giới tính: {info['gender']}\n"
        report += f"Vi khuẩn: {info['bacteria']}\n"
        report += f"Tiểu đường: {info['diabetes']}\n"
        report += f"Tăng huyết áp: {info['hypertension']}\n"
        report += f"Tiền sử nhập viện: {info['hospital_before']}\n"
        report += f"Tần suất nhiễm trùng: {info['infection_freq']}\n\n"

        resistance = explanation["resistance_predictions"]
        if resistance["sensitive"]:
            report += "Kháng sinh nhạy:\n"
            for ab in resistance["sensitive"]:
                report += f"  - {ab['antibiotic']}: {ab['probability']:.1%}\n"
        if resistance["resistant"]:
            report += "\nKháng sinh kháng:\n"
            for ab in resistance["resistant"]:
                report += f"  - {ab['antibiotic']}: {ab['resistance_probability']:.1%}\n"

        if explanation["key_factors"]:
            report += "\nYếu tố quan trọng:\n"
            for factor in explanation["key_factors"][:10]:
                value = factor.get("importance", factor.get("value", ""))
                report += f"  - {factor['feature']}: {value}\n"

        report += "\nTÓM TẮT:\n"
        report += explanation["report"]
        report += "\n" + "=" * 80 + "\n"
        return report


class ExplainabilityEvaluationAgent:
    """Agent 3: wrap ExplainabilityAgent with evaluation helpers and embedding."""

    def __init__(self, explainability_agent: Optional[ExplainabilityAgent] = None, use_embedding: bool = True):
        self.explainability_agent = explainability_agent or ExplainabilityAgent(use_embedding=use_embedding)

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
        if probabilities.empty:
            return {"mean_confidence": 0.0, "uncertain_fraction": 1.0}

        proba_row = probabilities.iloc[0]
        stats = {
            "mean_confidence": float(proba_row.mean()),
            "median_confidence": float(proba_row.median()),
        }
        uncertain_mask = (proba_row >= 0.4) & (proba_row <= 0.6)
        stats["uncertain_fraction"] = float(uncertain_mask.sum() / len(proba_row))

        if ground_truth is not None and not ground_truth.empty:
            gt_row = ground_truth.iloc[0]
            pred_labels = (proba_row >= 0.5).astype(int)
            stats["observed_accuracy"] = float((pred_labels == gt_row).mean())

        return stats







