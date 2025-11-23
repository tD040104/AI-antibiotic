"""
Preprocessing module - Data cleaning and feature engineering
Combines functionality from Agent 1 (Data Cleaner) and Agent 2 (Feature Engineer)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List


class DataCleaner:
    """Data cleaning and normalization"""
    
    def __init__(self):
        self.antibiotic_columns = [
            'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
            'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 
            'Furanes', 'colistine'
        ]
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize data"""
        df = df.copy()
        df = self._parse_age_gender(df)
        df = self._normalize_yes_no(df)
        df = self._normalize_dates(df)
        df = self._handle_missing_data(df)
        df = self._normalize_antibiotic_labels(df)
        df = self._normalize_infection_freq(df)
        df = self._remove_incomplete_rows(df)
        return df
    
    def _parse_age_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract age and gender from age/gender column"""
        def extract_age_gender(value):
            if pd.isna(value) or value == '':
                return None, None
            value_str = str(value)
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
        """Normalize Yes/No columns to binary"""
        yes_no_cols = ['Diabetes', 'Hypertension', 'Hospital_before']
        for col in yes_no_cols:
            if col in df.columns:
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
        """Normalize date columns"""
        def parse_date(date_str):
            if pd.isna(date_str) or str(date_str).lower() in ['error', 'nan', 'missing', '?', '']:
                return None
            date_str = str(date_str)
            date_str = date_str.replace('Fev', 'Feb').replace('Fév', 'Feb')
            formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d %b %Y', '%d %B %Y']
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
        if df['Collection_Date_parsed'].notna().any():
            max_date = df['Collection_Date_parsed'].max()
            df['Days_since_collection'] = (max_date - df['Collection_Date_parsed']).dt.days
            df['Days_since_collection'] = df['Days_since_collection'].fillna(0)
        else:
            df['Days_since_collection'] = 0
        return df
    
    def _normalize_infection_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Infection_Freq column"""
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
        """Normalize antibiotic labels S/R/I to binary"""
        for col in self.antibiotic_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper()
                replacements = {
                    'S': 'S', 'SENSITIVE': 'S', 'SENSITIF': 'S',
                    'R': 'R', 'RESISTANT': 'R', 'RESISTANTE': 'R',
                    'I': 'I', 'INTERMEDIATE': 'I',
                    'NAN': None, 'NONE': None, '?': None, 'ERROR': None, '': None
                }
                df[col] = df[col].replace(replacements)
                df[f'{col}_binary'] = (df[col] == 'S').astype(int)
                df[f'{col}_S'] = (df[col] == 'S').astype(int)
                df[f'{col}_R'] = (df[col] == 'R').astype(int)
                df[f'{col}_I'] = (df[col] == 'I').astype(int)
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data"""
        if 'Age' in df.columns:
            median_age = df['Age'].median()
            df['Age'] = df['Age'].fillna(median_age)
        if 'Gender' in df.columns:
            mode_gender = df['Gender'].mode()[0] if not df['Gender'].mode().empty else 'Female'
            df['Gender'] = df['Gender'].fillna(mode_gender)
        if 'Souches' in df.columns:
            df['Souches'] = df['Souches'].astype(str)
            df['Souches'] = df['Souches'].str.replace(r'^S\d+\s*', '', regex=True)
            df['Bacteria'] = df['Souches'].apply(self._normalize_bacteria_name)
        return df
    
    def _normalize_bacteria_name(self, name: str) -> str:
        """Normalize bacteria name"""
        if pd.isna(name) or str(name).lower() in ['none', 'nan', '']:
            return 'Unknown'
        name = str(name).strip()
        name = name.replace('E.coli', 'Escherichia coli')
        name = name.replace('E.coi', 'Escherichia coli')
        name = name.replace('E.cli', 'Escherichia coli')
        name = name.replace('Klbsiella', 'Klebsiella')
        name = name.replace('Klebsie.lla', 'Klebsiella')
        name = name.replace('Proeus', 'Proteus')
        name = name.replace('Prot.eus', 'Proteus')
        return name
    
    def _remove_incomplete_rows(self, df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """Remove rows with too much missing data"""
        available_antibiotic_cols = [col for col in self.antibiotic_columns if col in df.columns]
        if len(available_antibiotic_cols) > 0:
            valid_ratio = df[available_antibiotic_cols].notna().sum(axis=1) / len(available_antibiotic_cols)
            df = df[valid_ratio >= threshold].copy()
        return df


class FeatureEngineer:
    """Feature engineering for machine learning"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.bacteria_encoder = LabelEncoder()
        self.is_fitted = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning"""
        df = df.copy()
        df = self._encode_bacteria(df)
        df = self._create_clinical_features(df)
        df = self._normalize_infection_freq(df)
        df = self._create_interaction_features(df)
        return df
    
    def _encode_bacteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode bacteria type"""
        if 'Bacteria' not in df.columns:
            df['Bacteria'] = 'Unknown'
        bacteria_dummies = pd.get_dummies(df['Bacteria'], prefix='Bacteria')
        df = pd.concat([df, bacteria_dummies], axis=1)
        if not self.is_fitted:
            df['Bacteria_encoded'] = self.bacteria_encoder.fit_transform(df['Bacteria'])
        else:
            known_classes = set(self.bacteria_encoder.classes_)
            df['Bacteria_encoded'] = df['Bacteria'].apply(
                lambda x: self.bacteria_encoder.transform([x])[0] 
                if x in known_classes else len(known_classes)
            )
        return df
    
    def _create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create clinical features"""
        df['High_risk_diabetes_hospital'] = (
            (df.get('Diabetes', 0) == 1) & (df.get('Hospital_before', 0) == 1)
        ).astype(int)
        df['High_risk_hypertension_hospital'] = (
            (df.get('Hypertension', 0) == 1) & (df.get('Hospital_before', 0) == 1)
        ).astype(int)
        df['Total_risk_factors'] = (
            df.get('Diabetes', 0) + df.get('Hypertension', 0) + df.get('Hospital_before', 0)
        )
        if 'Age' in df.columns:
            df['Age_infection_interaction'] = df['Age'] * df.get('Infection_Freq', 0)
            df['Age_group'] = pd.cut(
                df['Age'], bins=[0, 18, 35, 50, 65, 100],
                labels=['Child', 'Young', 'Middle', 'Senior', 'Elderly']
            )
            age_dummies = pd.get_dummies(df['Age_group'], prefix='Age_group')
            df = pd.concat([df, age_dummies], axis=1)
        if 'Gender' in df.columns:
            df['Gender_encoded'] = df['Gender'].map({'Female': 0, 'Male': 1}).fillna(0)
            gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
            df = pd.concat([df, gender_dummies], axis=1)
        df['High_infection_freq'] = (df.get('Infection_Freq', 0) >= 2.0).astype(int)
        df['Risk_score'] = (
            df.get('Diabetes', 0) * 1.5 +
            df.get('Hypertension', 0) * 1.2 +
            df.get('Hospital_before', 0) * 2.0 +
            df.get('High_infection_freq', 0) * 1.8 +
            (df.get('Age', 0) / 100.0) * 1.0
        )
        return df
    
    def _normalize_infection_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize infection frequency"""
        if 'Infection_Freq' not in df.columns:
            df['Infection_Freq'] = 0.0
        df['Infection_Freq_log'] = np.log1p(df['Infection_Freq'])
        try:
            if self.is_fitted:
                scaled_values = self.scaler.transform(df[['Infection_Freq']].values)
                df['Infection_Freq_scaled'] = scaled_values.flatten()
            else:
                scaled_values = self.scaler.fit_transform(df[['Infection_Freq']].values)
                df['Infection_Freq_scaled'] = scaled_values.flatten()
                self.is_fitted = True
        except:
            df['Infection_Freq_scaled'] = df['Infection_Freq']
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        if 'Bacteria_encoded' in df.columns:
            df['Bacteria_risk_interaction'] = (
                df['Bacteria_encoded'] * df.get('Total_risk_factors', 0)
            )
        if 'Age' in df.columns:
            df['Age_infection_product'] = df['Age'] * df.get('Infection_Freq', 0)
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for training"""
        exclude_cols = [
            'ID', 'Name', 'Email', 'Address', 'age/gender', 'Souches',
            'Collection_Date', 'Collection_Date_parsed', 'Notes', 'Bacteria', 
            'Gender', 'Age_group', 'Days_since_collection'
        ]
        exclude_cols.extend([
            'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
            'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole',
            'Furanes', 'colistine'
        ])
        antibiotic_binary_cols = []
        for ab in [
            'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
            'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole',
            'Furanes', 'colistine'
        ]:
            antibiotic_binary_cols.extend([f'{ab}_binary', f'{ab}_S', f'{ab}_R', f'{ab}_I'])
        exclude_cols.extend(antibiotic_binary_cols)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols



