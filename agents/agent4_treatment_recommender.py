"""
Agent 4 - Treatment Recommender
Nhiệm vụ: Dựa trên dự đoán của Agent 3 để gợi ý thuốc điều trị
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class TreatmentRecommenderAgent:
    def __init__(self):
        # Ánh xạ tên kháng sinh sang tên đầy đủ
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
        
        # Phân loại kháng sinh theo nhóm (theo guideline)
        self.antibiotic_groups = {
            'Beta-lactams': ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO'],
            'Carbapenems': ['IPM'],
            'Aminoglycosides': ['GEN', 'AN'],
            'Quinolones': ['Acide nalidixique', 'ofx', 'CIP'],
            'Others': ['C', 'Co-trimoxazole', 'Furanes', 'colistine']
        }
        
        # Mức độ ưu tiên (cao = nên dùng trước)
        self.antibiotic_priority = {
            'IPM': 1,  # Carbapenem - dự trữ
            'AN': 2,
            'GEN': 3,
            'CIP': 4,
            'ofx': 5,
            'CTX/CRO': 6,
            'AMC': 7,
            'CZ': 8,
            'FOX': 9,
            'AMX/AMP': 10,
            'colistine': 1,  # Dự trữ
            'Co-trimoxazole': 11,
            'Acide nalidixique': 12,
            'C': 13,
            'Furanes': 14
        }
    
    def recommend_treatment(self, 
                          resistance_proba: pd.DataFrame,
                          patient_data: pd.Series = None,
                          top_k: int = 3) -> List[Dict]:
        """
        Gợi ý thuốc điều trị dựa trên xác suất kháng thuốc
        
        Args:
            resistance_proba: DataFrame với xác suất Sensitive cho mỗi kháng sinh
            patient_data: Thông tin bệnh nhân (tùy chọn)
            top_k: Số lượng thuốc được gợi ý
        
        Returns:
            List các dict chứa thông tin thuốc được gợi ý
        """
        recommendations = []
        
        # Lấy xác suất Sensitive cho mẫu đầu tiên
        if len(resistance_proba) > 0:
            proba = resistance_proba.iloc[0]
        else:
            return recommendations
        
        # Sắp xếp theo xác suất Sensitive (cao nhất trước)
        sorted_antibiotics = proba.sort_values(ascending=False)
        
        # Lọc các kháng sinh có xác suất Sensitive >= 0.5
        sensitive_antibiotics = sorted_antibiotics[sorted_antibiotics >= 0.5]
        
        # Nếu không có kháng sinh nào có xác suất >= 0.5, lấy top_k có xác suất cao nhất
        if len(sensitive_antibiotics) == 0:
            sensitive_antibiotics = sorted_antibiotics.head(top_k)
        
        # Áp dụng quy tắc y học (guideline)
        recommendations = self._apply_medical_guidelines(
            sensitive_antibiotics, 
            patient_data,
            top_k
        )
        
        return recommendations
    
    def _apply_medical_guidelines(self,
                                  sensitive_proba: pd.Series,
                                  patient_data: pd.Series = None,
                                  top_k: int = 3) -> List[Dict]:
        """
        Áp dụng quy tắc y học để chọn thuốc phù hợp
        """
        recommendations = []
        
        # Tạo danh sách ứng viên với điểm số
        candidates = []
        
        for antibiotic, proba in sensitive_proba.items():
            # Điểm cơ bản = xác suất Sensitive
            score = proba
            
            # Điều chỉnh theo mức độ ưu tiên (ưu tiên thấp = điểm cao hơn)
            if antibiotic in self.antibiotic_priority:
                priority_bonus = 1.0 / (self.antibiotic_priority[antibiotic] + 1)
                score += priority_bonus * 0.2  # Trọng số nhỏ
            
            # Điều chỉnh theo bệnh nhân (nếu có thông tin)
            if patient_data is not None:
                # Tránh carbapenems cho bệnh nhân không nặng (theo guideline)
                if antibiotic == 'IPM' and patient_data.get('Total_risk_factors', 0) < 2:
                    score *= 0.8  # Giảm điểm một chút
                
                # Ưu tiên nhóm aminoglycosides cho nhiễm trùng nặng
                if antibiotic in ['GEN', 'AN'] and patient_data.get('Total_risk_factors', 0) >= 2:
                    score *= 1.1
            
            candidates.append({
                'antibiotic': antibiotic,
                'full_name': self.antibiotic_names.get(antibiotic, antibiotic),
                'sensitive_probability': proba,
                'score': score,
                'group': self._get_antibiotic_group(antibiotic)
            })
        
        # Sắp xếp theo điểm số
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Chọn top_k, đảm bảo đa dạng nhóm kháng sinh
        selected = []
        groups_used = set()
        
        for candidate in candidates:
            if len(selected) >= top_k:
                break
            
            # Ưu tiên đa dạng nhóm (nhưng vẫn ưu tiên điểm cao)
            if candidate['group'] not in groups_used or len(groups_used) >= top_k:
                selected.append(candidate)
                groups_used.add(candidate['group'])
        
        # Nếu chưa đủ, lấy thêm từ danh sách
        while len(selected) < top_k and len(selected) < len(candidates):
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                    break
        
        # Format kết quả
        for idx, candidate in enumerate(selected[:top_k]):
            recommendations.append({
                'rank': idx + 1,
                'antibiotic_code': candidate['antibiotic'],
                'antibiotic_name': candidate['full_name'],
                'sensitive_probability': round(candidate['sensitive_probability'], 3),
                'confidence': self._get_confidence_level(candidate['sensitive_probability']),
                'group': candidate['group'],
                'recommendation': self._generate_recommendation_text(candidate)
            })
        
        return recommendations
    
    def _get_antibiotic_group(self, antibiotic: str) -> str:
        """Lấy nhóm kháng sinh"""
        for group, antibiotics in self.antibiotic_groups.items():
            if antibiotic in antibiotics:
                return group
        return 'Others'
    
    def _get_confidence_level(self, proba: float) -> str:
        """Xác định mức độ tin cậy"""
        if proba >= 0.8:
            return 'High'
        elif proba >= 0.6:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_recommendation_text(self, candidate: Dict) -> str:
        """Tạo text gợi ý"""
        proba = candidate['sensitive_probability']
        
        if proba >= 0.8:
            return f"Highly recommended - High sensitivity probability"
        elif proba >= 0.6:
            return f"Recommended - Moderate sensitivity probability"
        else:
            return f"Consider with caution - Lower sensitivity probability"
    
    def get_alternative_treatments(self,
                                   resistance_proba: pd.DataFrame,
                                   primary_treatment: str) -> List[Dict]:
        """
        Tìm các phương án điều trị thay thế nếu phương án chính không khả dụng
        """
        recommendations = self.recommend_treatment(resistance_proba, top_k=5)
        
        # Loại bỏ phương án chính
        alternatives = [r for r in recommendations 
                        if r['antibiotic_code'] != primary_treatment]
        
        return alternatives[:3]  # Trả về 3 phương án thay thế










