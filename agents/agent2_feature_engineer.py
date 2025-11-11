"""
Agent 2 - Feature Engineer
Nhiệm vụ: Biến dữ liệu vi khuẩn và bệnh lý thành đặc trưng học máy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List


class FeatureEngineerAgent:
    def __init__(self):
        self.scaler = StandardScaler()
        self.bacteria_encoder = LabelEncoder()
        self.is_fitted = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các đặc trưng cho học máy
        """
        df = df.copy()
        
        # Mã hóa loại vi khuẩn
        df = self._encode_bacteria(df)
        
        # Tạo biến lâm sàng
        df = self._create_clinical_features(df)
        
        # Chuẩn hóa Infection_Freq
        df = self._normalize_infection_freq(df)
        
        # Tạo các đặc trưng tương tác
        df = self._create_interaction_features(df)
        
        return df
    
    def _encode_bacteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mã hóa loại vi khuẩn (one-hot encoding)"""
        if 'Bacteria' not in df.columns:
            df['Bacteria'] = 'Unknown'
        
        # One-hot encoding cho loại vi khuẩn
        bacteria_dummies = pd.get_dummies(df['Bacteria'], prefix='Bacteria')
        df = pd.concat([df, bacteria_dummies], axis=1)
        
        # Label encoding (để dùng với một số mô hình)
        if not self.is_fitted:
            df['Bacteria_encoded'] = self.bacteria_encoder.fit_transform(df['Bacteria'])
        else:
            # Xử lý các giá trị mới chưa thấy trong training
            known_classes = set(self.bacteria_encoder.classes_)
            df['Bacteria_encoded'] = df['Bacteria'].apply(
                lambda x: self.bacteria_encoder.transform([x])[0] 
                if x in known_classes else len(known_classes)
            )
        
        return df
    
    def _create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo các biến lâm sàng phức tạp"""
        # Nguy cơ cao kháng thuốc: Tiểu đường + Nhập viện
        df['High_risk_diabetes_hospital'] = (
            (df.get('Diabetes', 0) == 1) & 
            (df.get('Hospital_before', 0) == 1)
        ).astype(int)
        
        # Nguy cơ cao: Tăng huyết áp + Nhập viện
        df['High_risk_hypertension_hospital'] = (
            (df.get('Hypertension', 0) == 1) & 
            (df.get('Hospital_before', 0) == 1)
        ).astype(int)
        
        # Tổng số yếu tố nguy cơ
        df['Total_risk_factors'] = (
            df.get('Diabetes', 0) + 
            df.get('Hypertension', 0) + 
            df.get('Hospital_before', 0)
        )
        
        # Tương tác giữa tuổi và nhiễm trùng
        if 'Age' in df.columns:
            df['Age_infection_interaction'] = df['Age'] * df.get('Infection_Freq', 0)
        
        # Phân loại nhóm tuổi
        if 'Age' in df.columns:
            df['Age_group'] = pd.cut(
                df['Age'], 
                bins=[0, 18, 35, 50, 65, 100],
                labels=['Child', 'Young', 'Middle', 'Senior', 'Elderly']
            )
            age_dummies = pd.get_dummies(df['Age_group'], prefix='Age_group')
            df = pd.concat([df, age_dummies], axis=1)
        
        # Mã hóa giới tính
        if 'Gender' in df.columns:
            df['Gender_encoded'] = df['Gender'].map({'Female': 0, 'Male': 1}).fillna(0)
            gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
            df = pd.concat([df, gender_dummies], axis=1)
        
        # Tần suất nhiễm trùng cao
        df['High_infection_freq'] = (df.get('Infection_Freq', 0) >= 2.0).astype(int)
        
        # Tính điểm nguy cơ tổng hợp
        df['Risk_score'] = (
            df.get('Diabetes', 0) * 1.5 +
            df.get('Hypertension', 0) * 1.2 +
            df.get('Hospital_before', 0) * 2.0 +
            df.get('High_infection_freq', 0) * 1.8 +
            (df.get('Age', 0) / 100.0) * 1.0
        )
        
        return df
    
    def _normalize_infection_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa số lần nhiễm trùng"""
        if 'Infection_Freq' not in df.columns:
            df['Infection_Freq'] = 0.0
        
        # Log transform để xử lý skewness
        df['Infection_Freq_log'] = np.log1p(df['Infection_Freq'])
        
        # Standard scaling (sẽ fit trong quá trình training)
        try:
            if self.is_fitted:
                scaled_values = self.scaler.transform(
                    df[['Infection_Freq']].values
                )
                df['Infection_Freq_scaled'] = scaled_values.flatten()
            else:
                scaled_values = self.scaler.fit_transform(
                    df[['Infection_Freq']].values
                )
                df['Infection_Freq_scaled'] = scaled_values.flatten()
                self.is_fitted = True
        except:
            # Nếu scaler chưa được fit hoặc có lỗi, sử dụng giá trị gốc
            df['Infection_Freq_scaled'] = df['Infection_Freq']
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo các đặc trưng tương tác giữa các biến"""
        # Tương tác giữa vi khuẩn và yếu tố nguy cơ
        if 'Bacteria_encoded' in df.columns:
            df['Bacteria_risk_interaction'] = (
                df['Bacteria_encoded'] * df.get('Total_risk_factors', 0)
            )
        
        # Tương tác giữa tuổi và tần suất nhiễm trùng
        if 'Age' in df.columns:
            df['Age_infection_product'] = (
                df['Age'] * df.get('Infection_Freq', 0)
            )
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Lấy danh sách các cột đặc trưng để dùng cho training"""
        exclude_cols = [
            'ID', 'Name', 'Email', 'Address', 'age/gender', 'Souches',
            'Collection_Date', 'Collection_Date_parsed', 'Notes', 'Bacteria', 
            'Gender', 'Age_group', 'Days_since_collection'
        ]
        
        # Loại bỏ các cột nhãn kháng sinh gốc và binary/one-hot encoding
        exclude_cols.extend([
            'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
            'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole',
            'Furanes', 'colistine'
        ])
        
        # Loại bỏ tất cả các cột binary và one-hot của kháng sinh (tránh data leakage)
        antibiotic_binary_cols = []
        for ab in [
            'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
            'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole',
            'Furanes', 'colistine'
        ]:
            antibiotic_binary_cols.extend([
                f'{ab}_binary',
                f'{ab}_S',
                f'{ab}_R',
                f'{ab}_I'
            ])
        
        exclude_cols.extend(antibiotic_binary_cols)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

