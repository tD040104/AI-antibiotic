"""
Agent 1 - Data Cleaner & Normalizer
Nhiệm vụ: Làm sạch dữ liệu đầu vào, chuẩn hóa các cột, xử lý dữ liệu thiếu
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, Tuple


class DataCleanerAgent:
    def __init__(self):
        self.antibiotic_columns = [
            'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
            'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 
            'Furanes', 'colistine'
        ]
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thực hiện làm sạch dữ liệu
        """
        df = df.copy()
        
        # Tách age và gender từ cột age/gender
        df = self._parse_age_gender(df)
        
        # Chuẩn hóa các cột Yes/No
        df = self._normalize_yes_no(df)
        
        # Chuẩn hóa ngày tháng
        df = self._normalize_dates(df)
        
        # Xử lý dữ liệu thiếu
        df = self._handle_missing_data(df)
        
        # Chuẩn hóa nhãn kháng sinh S/R/I
        df = self._normalize_antibiotic_labels(df)
        
        # Xử lý cột Infection_Freq
        df = self._normalize_infection_freq(df)
        
        # Loại bỏ các hàng có quá nhiều dữ liệu thiếu
        df = self._remove_incomplete_rows(df)
        
        return df
    
    def _parse_age_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tách tuổi và giới tính từ cột age/gender"""
        def extract_age_gender(value):
            if pd.isna(value) or value == '':
                return None, None
            
            value_str = str(value)
            # Tìm pattern số/tuổi và F/M
            age_match = re.search(r'(\d+)', value_str)
            gender_match = re.search(r'([FM])', value_str.upper())
            
            age = int(age_match.group(1)) if age_match else None
            gender = gender_match.group(1) if gender_match else None
            
            return age, gender
        
        df[['Age', 'Gender']] = df['age/gender'].apply(
            lambda x: pd.Series(extract_age_gender(x))
        )
        df['Gender'] = df['Gender'].map({'F': 'Female', 'M': 'Male'})
        
        return df
    
    def _normalize_yes_no(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa các cột Yes/No thành binary (0/1)"""
        yes_no_cols = ['Diabetes', 'Hypertension', 'Hospital_before']
        
        for col in yes_no_cols:
            if col in df.columns:
                # Chuẩn hóa các giá trị True/False/Yes/No
                # Sử dụng map để tránh FutureWarning về downcasting
                def normalize_value(val):
                    if pd.isna(val):
                        return 0
                    val_str = str(val).strip()
                    if val_str.upper() in ['YES', 'TRUE', '1', '1.0', True]:
                        return 1
                    elif val_str.upper() in ['NO', 'FALSE', '0', '0.0', False]:
                        return 0
                    else:
                        return 1 if val in [1, True] else 0
                
                df[col] = df[col].apply(normalize_value).astype(int)
        
        return df
    
    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa cột ngày tháng"""
        def parse_date(date_str):
            if pd.isna(date_str) or str(date_str).lower() in ['error', 'nan', 'missing', '?', '']:
                return None
            
            date_str = str(date_str)
            formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%d %b %Y',
                '%d %B %Y',
                '%d %b %Y',
            ]
            
            # Xử lý format đặc biệt như "5 Fev 2025" (tiếng Pháp)
            date_str = date_str.replace('Fev', 'Feb').replace('Fév', 'Feb')
            
            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
            
            try:
                return pd.to_datetime(date_str)
            except:
                return None
        
        df['Collection_Date_parsed'] = df['Collection_Date'].apply(parse_date)
        
        # Tính số ngày từ ngày thu thập đến hiện tại
        if df['Collection_Date_parsed'].notna().any():
            max_date = df['Collection_Date_parsed'].max()
            df['Days_since_collection'] = (max_date - df['Collection_Date_parsed']).dt.days
            df['Days_since_collection'] = df['Days_since_collection'].fillna(0)
        else:
            df['Days_since_collection'] = 0
        
        return df
    
    def _normalize_infection_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa cột Infection_Freq"""
        def normalize_freq(value):
            if pd.isna(value):
                return 0.0
            
            value_str = str(value).lower()
            if value_str in ['unknown', 'error', 'nan', 'missing', '?', '']:
                return 0.0
            
            try:
                return float(value)
            except:
                return 0.0
        
        df['Infection_Freq'] = df['Infection_Freq'].apply(normalize_freq)
        df['Infection_Freq'] = df['Infection_Freq'].fillna(0.0)
        
        return df
    
    def _normalize_antibiotic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa nhãn kháng sinh S/R/I thành binary hoặc one-hot"""
        # Chỉ xử lý nếu có các cột kháng sinh (khi training)
        # Khi predict cho bệnh nhân mới, các cột này không có
        for col in self.antibiotic_columns:
            if col in df.columns:
                # Chuẩn hóa về dạng S/R/I chuẩn
                df[col] = df[col].astype(str).str.upper()
                replacements = {
                    'S': 'S', 'SENSITIVE': 'S', 'SENSITIF': 'S',
                    'R': 'R', 'RESISTANT': 'R', 'RESISTANTE': 'R',
                    'I': 'I', 'INTERMEDIATE': 'I',
                    'NAN': None, 'NONE': None, '?': None, 'ERROR': None,
                    '': None
                }
                df[col] = df[col].replace(replacements)
                
                # Tạo binary encoding (S=1, R/I=0)
                df[f'{col}_binary'] = (df[col] == 'S').astype(int)
                
                # Tạo one-hot encoding
                df[f'{col}_S'] = (df[col] == 'S').astype(int)
                df[f'{col}_R'] = (df[col] == 'R').astype(int)
                df[f'{col}_I'] = (df[col] == 'I').astype(int)
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý dữ liệu thiếu"""
        # Điền giá trị thiếu cho Age bằng median
        if 'Age' in df.columns:
            median_age = df['Age'].median()
            df['Age'] = df['Age'].fillna(median_age)
        
        # Điền giá trị thiếu cho Gender bằng mode
        if 'Gender' in df.columns:
            mode_gender = df['Gender'].mode()[0] if not df['Gender'].mode().empty else 'Female'
            df['Gender'] = df['Gender'].fillna(mode_gender)
        
        # Xử lý Souches (tên vi khuẩn)
        if 'Souches' in df.columns:
            # Làm sạch và chuẩn hóa tên vi khuẩn
            df['Souches'] = df['Souches'].astype(str)
            # Loại bỏ ID prefix (ví dụ: S290 Escherichia coli -> Escherichia coli)
            df['Souches'] = df['Souches'].str.replace(r'^S\d+\s*', '', regex=True)
            df['Bacteria'] = df['Souches'].apply(self._normalize_bacteria_name)
        
        return df
    
    def _normalize_bacteria_name(self, name: str) -> str:
        """Chuẩn hóa tên vi khuẩn"""
        if pd.isna(name) or str(name).lower() in ['none', 'nan', '']:
            return 'Unknown'
        
        name = str(name).strip()
        # Chuẩn hóa các biến thể
        name = name.replace('E.coli', 'Escherichia coli')
        name = name.replace('E.coi', 'Escherichia coli')
        name = name.replace('E.cli', 'Escherichia coli')
        name = name.replace('Klbsiella', 'Klebsiella')
        name = name.replace('Klebsie.lla', 'Klebsiella')
        name = name.replace('Proeus', 'Proteus')
        name = name.replace('Prot.eus', 'Proteus')
        
        return name
    
    def _remove_incomplete_rows(self, df: pd.DataFrame, 
                                 threshold: float = 0.7) -> pd.DataFrame:
        """Loại bỏ các hàng có quá nhiều dữ liệu thiếu"""
        # Chỉ kiểm tra nếu các cột kháng sinh có trong DataFrame (training data)
        # Khi predict cho bệnh nhân mới, các cột này không có
        available_antibiotic_cols = [col for col in self.antibiotic_columns if col in df.columns]
        
        if len(available_antibiotic_cols) > 0:
            # Tính tỷ lệ dữ liệu có giá trị cho mỗi hàng
            valid_ratio = df[available_antibiotic_cols].notna().sum(axis=1) / len(available_antibiotic_cols)
            
            # Giữ lại các hàng có ít nhất threshold% dữ liệu hợp lệ
            df = df[valid_ratio >= threshold].copy()
        
        return df
    
    def get_cleaned_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trả về dữ liệu đã được làm sạch"""
        return self.clean(df)

