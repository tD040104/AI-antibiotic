"""
Prediction module - Main prediction pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import os
import joblib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import DataCleaner, FeatureEngineer
from src.modeling import ResistancePredictor


class Predictor:
    """Main prediction pipeline"""
    
    def __init__(self, model_path: str = None):
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.model = ResistancePredictor()
        self.feature_columns = None
        self.is_loaded = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str, state_path: str = None):
        """Load trained model and state"""
        # Load model
        if os.path.exists(model_path):
            self.model.load_model(model_path)
        
        # Load state if available
        if state_path and os.path.exists(state_path):
            state = joblib.load(state_path)
            self.feature_columns = state.get('feature_columns')
            self.data_cleaner = state.get('data_cleaner', self.data_cleaner)
            self.feature_engineer = state.get('feature_engineer', self.feature_engineer)
            if 'model_type' in state:
                self.model.model_type = state['model_type']
        
        self.is_loaded = True
    
    def predict(self, patient_data: dict) -> Dict:
        """
        Predict antibiotic resistance for a patient
        
        Args:
            patient_data: Dictionary with patient features:
                - age/gender: str (e.g., '45/F')
                - Souches: str (bacteria name)
                - Diabetes: str ('Yes' or 'No')
                - Hypertension: str ('Yes' or 'No')
                - Hospital_before: str ('Yes' or 'No')
                - Infection_Freq: float
                - Collection_Date: str
        
        Returns:
            Dictionary with predictions, probabilities, and resistance/sensitivity info
        """
        if not self.is_loaded:
            raise ValueError("Mô hình chưa được tải! Vui lòng load model trước.")
        
        try:
            # Convert to DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Clean data
            patient_cleaned = self.data_cleaner.clean(patient_df)
            
            # Feature engineering
            patient_features = self.feature_engineer.engineer_features(patient_cleaned)
            
            # Ensure all feature columns exist
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in patient_features.columns:
                        patient_features[col] = 0
                
                X_patient = patient_features[self.feature_columns].fillna(0)
            else:
                # Fallback: use all numeric columns
                numeric_cols = patient_features.select_dtypes(include=[np.number]).columns
                X_patient = patient_features[numeric_cols].fillna(0)
            
            # Ensure all columns are numeric
            X_patient = X_patient.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Predict
            predictions, probabilities = self.model.predict(X_patient)
            
            # Format results
            pred_dict = predictions.iloc[0].to_dict()
            proba_dict = probabilities.iloc[0].to_dict()
            
            # Create resistance/sensitivity summary
            resistance_info = self._create_resistance_summary(pred_dict, proba_dict)
            
            return {
                'predictions': pred_dict,
                'probabilities': proba_dict,
                'resistance_info': resistance_info,
                'patient_features': patient_features.iloc[0].to_dict()
            }
        except Exception as e:
            print(f"Lỗi trong quá trình predict: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_resistance_summary(self, predictions: Dict, probabilities: Dict) -> Dict:
        """Create summary of resistance/sensitivity information"""
        sensitive = []
        resistant = []
        
        antibiotic_names = {
            'AMX/AMP': 'Amoxicillin/Ampicillin',
            'AMC': 'Amoxicillin-Clavulanic Acid',
            'CZ': 'Cefazolin',
            'FOX': 'Cefoxitin',
            'CTX/CRO': 'Ceftriaxone/Cefotaxime',
            'IPM': 'Imipenem',
            'GEN': 'Gentamicin',
            'AN': 'Amikacin',
            'Acide nalidixique': 'Nalidixic Acid',
            'ofx': 'Ofloxacin',
            'CIP': 'Ciprofloxacin',
            'C': 'Chloramphenicol',
            'Co-trimoxazole': 'Trimethoprim-Sulfamethoxazole',
            'Furanes': 'Nitrofurantoin',
            'colistine': 'Colistin'
        }
        
        for ab_code, pred in predictions.items():
            proba = probabilities.get(ab_code, 0.0)
            ab_name = antibiotic_names.get(ab_code, ab_code)
            
            if pred == 1:  # Sensitive
                sensitive.append({
                    'code': ab_code,
                    'name': ab_name,
                    'sensitivity_probability': round(proba, 3),
                    'status': 'Sensitive'
                })
            else:  # Resistant
                resistant.append({
                    'code': ab_code,
                    'name': ab_name,
                    'resistance_probability': round(1 - proba, 3),
                    'status': 'Resistant'
                })
        
        return {
            'sensitive': sensitive,
            'resistant': resistant,
            'sensitive_count': len(sensitive),
            'resistant_count': len(resistant)
        }

