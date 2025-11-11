"""
Agent 5 - Explainability & Report Generator
Nhiệm vụ: Giải thích kết quả và tạo báo cáo bằng ngôn ngữ tự nhiên
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP không được cài đặt. Sẽ không sử dụng SHAP values.")


class ExplainabilityAgent:
    def __init__(self):
        self.antibiotic_names = {
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
    
    def explain_prediction(self,
                          patient_data: pd.Series,
                          predictions: pd.DataFrame,
                          probabilities: pd.DataFrame,
                          feature_importance: pd.DataFrame = None,
                          shap_values: np.ndarray = None,
                          feature_names: List[str] = None) -> Dict:
        """
        Giải thích dự đoán cho một bệnh nhân cụ thể
        """
        explanation = {
            'patient_summary': self._extract_patient_info(patient_data),
            'resistance_predictions': self._summarize_resistance(predictions.iloc[0], probabilities.iloc[0]),
            'key_factors': self._identify_key_factors(patient_data, feature_importance, shap_values, feature_names),
            'report': self._generate_natural_language_report(patient_data, predictions.iloc[0], probabilities.iloc[0])
        }
        
        return explanation
    
    def _extract_patient_info(self, patient_data: pd.Series) -> Dict:
        """Trích xuất thông tin bệnh nhân"""
        info = {
            'age': patient_data.get('Age', 'Unknown'),
            'gender': patient_data.get('Gender', 'Unknown'),
            'bacteria': patient_data.get('Bacteria', 'Unknown'),
            'diabetes': 'Có' if patient_data.get('Diabetes', 0) == 1 else 'Không',
            'hypertension': 'Có' if patient_data.get('Hypertension', 0) == 1 else 'Không',
            'hospital_before': 'Có' if patient_data.get('Hospital_before', 0) == 1 else 'Không',
            'infection_freq': patient_data.get('Infection_Freq', 0)
        }
        return info
    
    def _summarize_resistance(self, predictions: pd.Series, probabilities: pd.Series) -> Dict:
        """Tóm tắt dự đoán kháng thuốc"""
        resistant = []
        sensitive = []
        intermediate = []
        
        for antibiotic in predictions.index:
            pred = predictions[antibiotic]
            proba = probabilities[antibiotic]
            
            if pred == 1:  # Sensitive
                sensitive.append({
                    'antibiotic': self.antibiotic_names.get(antibiotic, antibiotic),
                    'code': antibiotic,
                    'probability': round(proba, 3)
                })
            else:  # Resistant/Intermediate
                resistant.append({
                    'antibiotic': self.antibiotic_names.get(antibiotic, antibiotic),
                    'code': antibiotic,
                    'resistance_probability': round(1 - proba, 3)
                })
        
        return {
            'sensitive': sensitive,
            'resistant': resistant,
            'sensitive_count': len(sensitive),
            'resistant_count': len(resistant)
        }
    
    def _identify_key_factors(self,
                              patient_data: pd.Series,
                              feature_importance: pd.DataFrame = None,
                              shap_values: np.ndarray = None,
                              feature_names: List[str] = None) -> List[Dict]:
        """Xác định các yếu tố chính ảnh hưởng đến dự đoán"""
        factors = []
        
        # Phân tích từ SHAP values nếu có
        if SHAP_AVAILABLE and shap_values is not None and feature_names is not None:
            # Lấy SHAP values cho mẫu này
            if len(shap_values.shape) > 2:  # Multi-output
                shap_values_sample = shap_values[:, 0, :]  # Lấy output đầu tiên
                shap_sum = np.abs(shap_values_sample).mean(axis=0)
            else:
                shap_sum = np.abs(shap_values).mean(axis=0)
            
            # Sắp xếp theo tầm quan trọng
            top_indices = np.argsort(shap_sum)[::-1][:10]
            
            for idx in top_indices:
                if idx < len(feature_names):
                    factors.append({
                        'feature': feature_names[idx],
                        'importance': float(shap_sum[idx]),
                        'type': 'shap'
                    })
        elif feature_importance is not None:
            # Sử dụng feature importance từ mô hình
            top_features = feature_importance.head(10)
            for _, row in top_features.iterrows():
                factors.append({
                    'feature': row['feature'],
                    'importance': float(row['importance']),
                    'type': 'model_importance'
                })
        
        # Thêm các yếu tố lâm sàng quan trọng
        clinical_factors = {
            'Diabetes': patient_data.get('Diabetes', 0),
            'Hypertension': patient_data.get('Hypertension', 0),
            'Hospital_before': patient_data.get('Hospital_before', 0),
            'Infection_Freq': patient_data.get('Infection_Freq', 0),
            'Total_risk_factors': patient_data.get('Total_risk_factors', 0),
            'Age': patient_data.get('Age', None)
        }
        
        for factor, value in clinical_factors.items():
            if value and value != 0:
                factors.append({
                    'feature': factor,
                    'value': value,
                    'type': 'clinical'
                })
        
        return factors[:15]  # Top 15 factors
    
    def _generate_natural_language_report(self,
                                          patient_data: pd.Series,
                                          predictions: pd.Series,
                                          probabilities: pd.Series) -> str:
        """Tạo báo cáo bằng ngôn ngữ tự nhiên"""
        # Thông tin bệnh nhân
        age = patient_data.get('Age', 'Unknown')
        gender = patient_data.get('Gender', 'Unknown')
        if gender == 'Female':
            gender_vn = 'nữ'
        elif gender == 'Male':
            gender_vn = 'nam'
        else:
            gender_vn = 'không xác định'
        
        bacteria = patient_data.get('Bacteria', 'Unknown')
        
        # Xác định các yếu tố nguy cơ
        risk_factors = []
        if patient_data.get('Diabetes', 0) == 1:
            risk_factors.append('tiểu đường')
        if patient_data.get('Hypertension', 0) == 1:
            risk_factors.append('tăng huyết áp')
        if patient_data.get('Hospital_before', 0) == 1:
            risk_factors.append('tiền sử nhập viện')
        
        risk_text = ', '.join(risk_factors) if risk_factors else 'không có yếu tố nguy cơ đáng kể'
        
        # Xác định kháng sinh nhạy và kháng
        sensitive_antibiotics = []
        resistant_antibiotics = []
        
        for antibiotic in predictions.index:
            if predictions[antibiotic] == 1:
                sensitive_antibiotics.append(self.antibiotic_names.get(antibiotic, antibiotic))
            else:
                if probabilities[antibiotic] < 0.3:  # Xác suất nhạy thấp = kháng cao
                    resistant_antibiotics.append(self.antibiotic_names.get(antibiotic, antibiotic))
        
        # Tạo báo cáo
        report = f"Bệnh nhân {gender_vn}, {age} tuổi, nhiễm {bacteria}"
        
        if risk_factors:
            report += f", có {risk_text}"
        
        report += ".\n\n"
        
        if resistant_antibiotics:
            report += f"Dự đoán kháng với: {', '.join(resistant_antibiotics[:5])}.\n"
        
        if sensitive_antibiotics:
            report += f"Dự đoán nhạy với: {', '.join(sensitive_antibiotics[:5])}.\n"
        
        # Thêm khuyến nghị
        if len(sensitive_antibiotics) > 0:
            try:
                sensitive_idx = predictions[predictions == 1].index
                if len(sensitive_idx) > 0:
                    first_sensitive = sensitive_idx[0]
                    proba_value = probabilities.get(first_sensitive, 0.0)
                    report += f"\nKhuyến nghị: Xem xét sử dụng {sensitive_antibiotics[0]} " \
                             f"(xác suất nhạy: {proba_value:.1%})."
                else:
                    report += f"\nKhuyến nghị: Xem xét sử dụng {sensitive_antibiotics[0]}."
            except:
                report += f"\nKhuyến nghị: Xem xét sử dụng {sensitive_antibiotics[0]}."
        else:
            report += "\nCảnh báo: Tất cả các kháng sinh được kiểm tra đều có khả năng kháng cao. " \
                     "Cần xem xét các phương án điều trị đặc biệt."
        
        return report
    
    def generate_detailed_report(self, explanation: Dict) -> str:
        """Tạo báo cáo chi tiết"""
        report = "=" * 80 + "\n"
        report += "BÁO CÁO PHÂN TÍCH KHÁNG KHÁNG SINH\n"
        report += "=" * 80 + "\n\n"
        
        # Thông tin bệnh nhân
        report += "THÔNG TIN BỆNH NHÂN:\n"
        report += "-" * 80 + "\n"
        info = explanation['patient_summary']
        report += f"Tuổi: {info['age']}\n"
        report += f"Giới tính: {info['gender']}\n"
        report += f"Vi khuẩn: {info['bacteria']}\n"
        report += f"Tiểu đường: {info['diabetes']}\n"
        report += f"Tăng huyết áp: {info['hypertension']}\n"
        report += f"Tiền sử nhập viện: {info['hospital_before']}\n"
        report += f"Tần suất nhiễm trùng: {info['infection_freq']}\n\n"
        
        # Dự đoán kháng thuốc
        report += "DỰ ĐOÁN KHÁNG THUỐC:\n"
        report += "-" * 80 + "\n"
        resistance = explanation['resistance_predictions']
        
        if resistance['sensitive']:
            report += "\nKháng sinh nhạy (Sensitive):\n"
            for ab in resistance['sensitive']:
                report += f"  - {ab['antibiotic']}: {ab['probability']:.1%}\n"
        
        if resistance['resistant']:
            report += "\nKháng sinh kháng (Resistant):\n"
            for ab in resistance['resistant']:
                report += f"  - {ab['antibiotic']}: {ab['resistance_probability']:.1%}\n"
        
        # Yếu tố quan trọng
        if explanation['key_factors']:
            report += "\nYẾU TỐ QUAN TRỌNG:\n"
            report += "-" * 80 + "\n"
            for factor in explanation['key_factors'][:10]:
                report += f"  - {factor['feature']}: {factor.get('importance', factor.get('value', ''))}\n"
        
        # Báo cáo tự nhiên
        report += "\nTÓM TẮT:\n"
        report += "-" * 80 + "\n"
        report += explanation['report']
        
        report += "\n\n" + "=" * 80 + "\n"
        
        return report

