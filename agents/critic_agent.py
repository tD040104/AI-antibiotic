"""
Critic agent - detect uncertain predictions or missing data
Enhanced with Embedding and Transformer for decision making
"""

from __future__ import annotations

from dataclasses import dataclass
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
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None  # type: ignore


@dataclass
class CriticFlag:
    antibiotic: str
    probability: float
    reason: str


class CriticEmbedder:
    """Embedding layer for critic agent using transformer architecture."""
    
    def __init__(self, input_dim: int = 30, embedding_dim: int = 128, hidden_dim: int = 256):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CriticEmbedder. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Decision head for uncertainty detection
        self.uncertainty_head = nn.Sequential(
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
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)
        
        embedded = self.embedding_layer(X_tensor.squeeze(1))
        embedded = embedded.unsqueeze(1)
        encoded = self.transformer(embedded)
        return encoded.squeeze(1)
    
    def predict_uncertainty(self, X: np.ndarray) -> float:
        """Predict uncertainty score using transformer."""
        with torch.no_grad():
            encoded = self.transform(X)
            uncertainty_score = self.uncertainty_head(encoded)
            return uncertainty_score.item()


class CriticAgent:
    """
    Agent 4: flag low-confidence predictions or incomplete patient data.
    Enhanced with embedding and transformer for decision making.
    """

    def __init__(self, uncertainty_band: Tuple[float, float] = (0.4, 0.6), use_embedding: bool = True):
        self.uncertainty_band = uncertainty_band
        self.use_embedding = use_embedding and TORCH_AVAILABLE
        self.embedder = None
        
        if self.use_embedding:
            try:
                self.embedder = CriticEmbedder(input_dim=30, embedding_dim=128)
            except Exception:
                self.embedder = None
                self.use_embedding = False

    def _create_critic_embedding(
        self, 
        probabilities: pd.Series, 
        patient_features: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Create embedding vector for critic analysis."""
        feature_values = []
        
        # Add probabilities
        for ab in probabilities.index:
            feature_values.append(float(probabilities[ab]))
        
        # Add patient features if available
        if patient_features is not None:
            numeric_features = ['Age', 'Diabetes', 'Hypertension', 'Hospital_before', 
                               'Infection_Freq', 'Total_risk_factors']
            for feat in numeric_features:
                val = patient_features.get(feat, 0)
                feature_values.append(float(val) if not pd.isna(val) else 0.0)
        
        # Pad or truncate to fixed size
        target_size = 30
        if len(feature_values) < target_size:
            feature_values.extend([0.0] * (target_size - len(feature_values)))
        else:
            feature_values = feature_values[:target_size]
        
        return np.array(feature_values).reshape(1, -1)
    
    def get_embedding_vector(
        self,
        probabilities: pd.Series,
        patient_features: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """Get embedding vector representation for machine learning models.
        
        Returns:
            numpy array of shape (embedding_dim,) containing the critic embedding
        """
        if self.use_embedding and self.embedder is not None:
            try:
                # Create feature vector
                feature_vector = self._create_critic_embedding(probabilities, patient_features)
                
                # Get transformer embedding
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        embedding = self.embedder.transform(feature_vector)
                        # Return as numpy array
                        return embedding.cpu().numpy().flatten()
                else:
                    # Fallback: return raw feature vector
                    return feature_vector.flatten()
            except Exception:
                # Fallback: return raw feature vector
                feature_vector = self._create_critic_embedding(probabilities, patient_features)
                return feature_vector.flatten()
        else:
            # Return raw feature vector without transformer
            feature_vector = self._create_critic_embedding(probabilities, patient_features)
            return feature_vector.flatten()
    
    def review_vector(
        self,
        probabilities: pd.DataFrame,
        patient_features: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """Return vector representation for Agent 5 machine learning models.
        
        This method returns a numerical vector (array/tensor) instead of dict,
        which can be directly fed into Decision Tree, Fuzzy Logic, LLM embedding, etc.
        
        Returns:
            numpy array of shape (embedding_dim,) or (30,) containing embedding vector.
            - If transformer is available: shape (embedding_dim,) = (128,)
            - If transformer is not available: shape (30,) = raw feature vector
            - If probabilities is empty: returns zero vector of appropriate size
        """
        if probabilities.empty:
            # Return zero vector if no probabilities
            return np.zeros(128 if self.use_embedding and self.embedder else 30)
        
        return self.get_embedding_vector(probabilities.iloc[0], patient_features)
    
    def _generate_decision_from_embedding(
        self,
        probabilities: pd.Series,
        patient_features: Optional[pd.Series] = None
    ) -> Dict:
        """Generate decision using transformer-based embedding."""
        if not self.use_embedding or self.embedder is None:
            return self._generate_rule_based_decision(probabilities, patient_features)
        
        try:
            # Create embedding
            feature_vector = self._create_critic_embedding(probabilities, patient_features)
            
            # Get uncertainty score from transformer
            uncertainty_score = self.embedder.predict_uncertainty(feature_vector)
            
            # Generate decision based on uncertainty
            if uncertainty_score >= 0.6:
                decision_type = "high_uncertainty_requires_testing"
                confidence = "low"
            elif uncertainty_score >= 0.3:
                decision_type = "moderate_uncertainty_review_needed"
                confidence = "moderate"
            else:
                decision_type = "low_uncertainty_confident"
                confidence = "high"
            
            return {
                "decision_type": decision_type,
                "uncertainty_score": float(uncertainty_score),
                "confidence": confidence,
                "reasoning": f"Transformer-based uncertainty score: {uncertainty_score:.3f}"
            }
        except Exception as e:
            return self._generate_rule_based_decision(probabilities, patient_features)
    
    def _generate_rule_based_decision(
        self,
        probabilities: pd.Series,
        patient_features: Optional[pd.Series] = None
    ) -> Dict:
        """Fallback rule-based decision."""
        mean_prob = probabilities.mean()
        std_prob = probabilities.std()
        
        if std_prob > 0.3 or (mean_prob > 0.4 and mean_prob < 0.6):
            decision_type = "high_uncertainty_requires_testing"
            confidence = "low"
        elif std_prob > 0.2:
            decision_type = "moderate_uncertainty_review_needed"
            confidence = "moderate"
        else:
            decision_type = "low_uncertainty_confident"
            confidence = "high"
        
        return {
            "decision_type": decision_type,
            "uncertainty_score": float(std_prob),
            "confidence": confidence,
            "reasoning": "Rule-based fallback decision"
        }

    def review(
        self,
        probabilities: pd.DataFrame,
        patient_features: Optional[pd.Series] = None,
        missing_required_fields: Optional[List[str]] = None,
    ) -> Dict:
        if probabilities.empty:
            return {
                "flags": [], 
                "missing_fields": missing_required_fields or [],
                "decision": {
                    "decision_type": "insufficient_data",
                    "uncertainty_score": 1.0,
                    "confidence": "low",
                    "reasoning": "No probability data available"
                }
            }

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

        # Generate decision using embedding/transformer
        decision = self._generate_decision_from_embedding(proba_row, patient_features)

        return {
            "flags": flags,
            "missing_fields": sorted(set(missing)),
            "needs_additional_tests": bool(flags or missing),
            "decision": decision,  # Add decision output
        }

