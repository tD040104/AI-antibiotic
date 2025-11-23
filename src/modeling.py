"""
Modeling module - Machine learning model for antibiotic resistance prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from typing import Dict, List, Tuple
import joblib
import os


class ResistancePredictor:
    """Machine learning model for predicting antibiotic resistance"""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        model_type: 'xgboost', 'random_forest', or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.antibiotic_columns = [
            'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
            'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole',
            'Furanes', 'colistine'
        ]
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.DataFrame, 
              test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if self.model_type == 'xgboost':
            base_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        else:  # gradient_boosting
            base_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            )
        
        self.model = MultiOutputClassifier(base_model, n_jobs=-1)
        
        print(f"Đang huấn luyện mô hình {self.model_type}...")
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_proba = self.model.predict_proba(X_train)
        test_proba = self.model.predict_proba(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_auc = {}
        test_auc = {}
        
        for idx, col in enumerate(y.columns):
            try:
                train_auc[col] = roc_auc_score(
                    y_train.iloc[:, idx], 
                    train_proba[idx][:, 1]
                )
                test_auc[col] = roc_auc_score(
                    y_test.iloc[:, idx],
                    test_proba[idx][:, 1]
                )
            except:
                train_auc[col] = None
                test_auc[col] = None
        
        self.is_trained = True
        
        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'model_type': self.model_type
        }
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predict antibiotic resistance"""
        if not self.is_trained:
            raise ValueError("Mô hình chưa được huấn luyện!")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        pred_df = pd.DataFrame(
            predictions,
            columns=self.antibiotic_columns,
            index=X.index
        )
        
        proba_dict = {}
        for idx, col in enumerate(self.antibiotic_columns):
            if idx < len(probabilities):
                proba_dict[col] = probabilities[idx][:, 1]  # Sensitive probability
        
        proba_df = pd.DataFrame(proba_dict, index=X.index)
        
        return pred_df, proba_df
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only probabilities"""
        _, proba_df = self.predict(X)
        return proba_df
    
    def save_model(self, filepath: str):
        """Save the model"""
        if self.model is None:
            raise ValueError("Không có mô hình để lưu!")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Mô hình đã được lưu tại: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Mô hình đã được tải từ: {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Mô hình chưa được huấn luyện!")
        
        feature_names = None
        importance_sum = None
        
        for estimator in self.model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(len(estimator.feature_importances_))]
                    importance_sum = np.zeros(len(estimator.feature_importances_))
                importance_sum += estimator.feature_importances_
        
        if importance_sum is not None:
            importance_avg = importance_sum / len(self.model.estimators_)
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance_avg
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame()



