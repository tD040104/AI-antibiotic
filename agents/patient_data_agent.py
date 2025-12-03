"""
PatientDataAgent - ingest + preprocess patient data for MAS pipeline
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import re


RecordType = Union[Dict, List[Dict], pd.Series, pd.DataFrame]

ANTIBIOTIC_CODES = [
    "AMX/AMP",
    "AMC",
    "CZ",
    "FOX",
    "CTX/CRO",
    "IPM",
    "GEN",
    "AN",
    "Acide nalidixique",
    "ofx",
    "CIP",
    "C",
    "Co-trimoxazole",
    "Furanes",
    "colistine",
]


class DataCleaner:
    """Data cleaning and normalization logic."""

    def __init__(self):
        self.antibiotic_columns = ANTIBIOTIC_CODES

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
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
        def extract_age_gender(value):
            if pd.isna(value) or value == "":
                return None, None
            value_str = str(value)
            age_match = re.search(r"(\d+)", value_str)
            gender_match = re.search(r"([FM])", value_str.upper())
            age = int(age_match.group(1)) if age_match else None
            gender = gender_match.group(1) if gender_match else None
            return age, gender

        if "age/gender" in df.columns:
            df[["Age", "Gender"]] = df["age/gender"].apply(
                lambda x: pd.Series(extract_age_gender(x))
            )
            df["Gender"] = df["Gender"].map({"F": "Female", "M": "Male"})
        else:
            if "Age" not in df.columns:
                df["Age"] = None
            if "Gender" not in df.columns:
                df["Gender"] = None
        return df

    def _normalize_yes_no(self, df: pd.DataFrame) -> pd.DataFrame:
        yes_no_cols = ["Diabetes", "Hypertension", "Hospital_before"]
        for col in yes_no_cols:
            if col in df.columns:
                def normalize_value(val):
                    if pd.isna(val):
                        return 0
                    val_str = str(val).strip()
                    if val_str.upper() in ["YES", "TRUE", "1", "1.0", True]:
                        return 1
                    if val_str.upper() in ["NO", "FALSE", "0", "0.0", False]:
                        return 0
                    return 1 if val in [1, True] else 0

                df[col] = df[col].apply(normalize_value).astype(int)
        return df

    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        def parse_date(date_str):
            if pd.isna(date_str) or str(date_str).lower() in ["error", "nan", "missing", "?", ""]:
                return None
            date_str = str(date_str).replace("Fev", "Feb").replace("Fév", "Feb")
            formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%d %B %Y"]
            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except Exception:
                    continue
            try:
                return pd.to_datetime(date_str)
            except Exception:
                return None

        if "Collection_Date" in df.columns:
            df["Collection_Date_parsed"] = df["Collection_Date"].apply(parse_date)
            if df["Collection_Date_parsed"].notna().any():
                max_date = df["Collection_Date_parsed"].max()
                df["Days_since_collection"] = (max_date - df["Collection_Date_parsed"]).dt.days
                df["Days_since_collection"] = df["Days_since_collection"].fillna(0)
            else:
                df["Days_since_collection"] = 0
        else:
            df["Collection_Date_parsed"] = None
            df["Days_since_collection"] = 0
        return df

    def _normalize_infection_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        def normalize_freq(value):
            if pd.isna(value):
                return 0.0
            value_str = str(value).lower()
            if value_str in ["unknown", "error", "nan", "missing", "?", ""]:
                return 0.0
            try:
                return float(value)
            except Exception:
                return 0.0

        if "Infection_Freq" in df.columns:
            df["Infection_Freq"] = df["Infection_Freq"].apply(normalize_freq)
            df["Infection_Freq"] = df["Infection_Freq"].fillna(0.0)
        else:
            df["Infection_Freq"] = 0.0
        return df

    def _normalize_antibiotic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.antibiotic_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper()
                replacements = {
                    "S": "S", "SENSITIVE": "S", "SENSITIF": "S",
                    "R": "R", "RESISTANT": "R", "RESISTANTE": "R",
                    "I": "I", "INTERMEDIATE": "I",
                    "NAN": None, "NONE": None, "?": None, "ERROR": None, "": None
                }
                df[col] = df[col].replace(replacements)
                df[f"{col}_binary"] = (df[col] == "S").astype(int)
                df[f"{col}_S"] = (df[col] == "S").astype(int)
                df[f"{col}_R"] = (df[col] == "R").astype(int)
                df[f"{col}_I"] = (df[col] == "I").astype(int)
        return df

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Age" in df.columns:
            median_age = df["Age"].median()
            df["Age"] = df["Age"].fillna(median_age)
        if "Gender" in df.columns:
            mode_gender = df["Gender"].mode()[0] if not df["Gender"].mode().empty else "Female"
            df["Gender"] = df["Gender"].fillna(mode_gender)
        if "Souches" in df.columns:
            df["Souches"] = df["Souches"].astype(str)
            df["Souches"] = df["Souches"].str.replace(r"^S\d+\s*", "", regex=True)
            df["Bacteria"] = df["Souches"].apply(self._normalize_bacteria_name)
        return df

    def _normalize_bacteria_name(self, name: str) -> str:
        if pd.isna(name) or str(name).lower() in ["none", "nan", ""]:
            return "Unknown"
        name = str(name).strip()
        replacements = {
            "E.coli": "Escherichia coli",
            "E.coi": "Escherichia coli",
            "E.cli": "Escherichia coli",
            "Klbsiella": "Klebsiella",
            "Klebsie.lla": "Klebsiella",
            "Proeus": "Proteus",
            "Prot.eus": "Proteus",
        }
        for old, new in replacements.items():
            name = name.replace(old, new)
        return name

    def _remove_incomplete_rows(self, df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        available = [col for col in self.antibiotic_columns if col in df.columns]
        if available:
            valid_ratio = df[available].notna().sum(axis=1) / len(available)
            df = df[valid_ratio >= threshold].copy()
        return df


class FeatureEngineer:
    """Feature engineering logic for machine learning."""

    def __init__(self):
        self.scaler = None
        self.bacteria_encoder = None
        self.is_fitted = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._encode_bacteria(df)
        df = self._create_clinical_features(df)
        df = self._normalize_infection_freq(df)
        df = self._create_interaction_features(df)
        return df

    def _encode_bacteria(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Bacteria" not in df.columns:
            df["Bacteria"] = "Unknown"
        bacteria_dummies = pd.get_dummies(df["Bacteria"], prefix="Bacteria")
        df = pd.concat([df, bacteria_dummies], axis=1)

        if self.bacteria_encoder is None:
            from sklearn.preprocessing import LabelEncoder

            self.bacteria_encoder = LabelEncoder()
            df["Bacteria_encoded"] = self.bacteria_encoder.fit_transform(df["Bacteria"])
        else:
            known_classes = set(self.bacteria_encoder.classes_)
            df["Bacteria_encoded"] = df["Bacteria"].apply(
                lambda x: self.bacteria_encoder.transform([x])[0]
                if x in known_classes else len(known_classes)
            )
        return df

    def _create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Diabetes"] = df.get("Diabetes", 0)
        df["Hypertension"] = df.get("Hypertension", 0)
        df["Hospital_before"] = df.get("Hospital_before", 0)
        df["Infection_Freq"] = df.get("Infection_Freq", 0.0)

        df["High_risk_diabetes_hospital"] = (
            (df["Diabetes"] == 1) & (df["Hospital_before"] == 1)
        ).astype(int)
        df["High_risk_hypertension_hospital"] = (
            (df["Hypertension"] == 1) & (df["Hospital_before"] == 1)
        ).astype(int)
        df["Total_risk_factors"] = df["Diabetes"] + df["Hypertension"] + df["Hospital_before"]

        if "Age" in df.columns:
            df["Age_infection_interaction"] = df["Age"] * df["Infection_Freq"]
            df["Age_group"] = pd.cut(
                df["Age"], bins=[0, 18, 35, 50, 65, 100],
                labels=["Child", "Young", "Middle", "Senior", "Elderly"]
            )
            age_dummies = pd.get_dummies(df["Age_group"], prefix="Age_group")
            df = pd.concat([df, age_dummies], axis=1)

        if "Gender" in df.columns:
            df["Gender_encoded"] = df["Gender"].map({"Female": 0, "Male": 1}).fillna(0)
            gender_dummies = pd.get_dummies(df["Gender"], prefix="Gender")
            df = pd.concat([df, gender_dummies], axis=1)

        df["High_infection_freq"] = (df["Infection_Freq"] >= 2.0).astype(int)
        df["Age"] = df.get("Age", 0)
        df["Risk_score"] = (
            df["Diabetes"] * 1.5 +
            df["Hypertension"] * 1.2 +
            df["Hospital_before"] * 2.0 +
            df["High_infection_freq"] * 1.8 +
            (df["Age"] / 100.0)
        )
        return df

    def _normalize_infection_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Infection_Freq"] = df.get("Infection_Freq", 0.0)
        df["Infection_Freq_log"] = np.log1p(df["Infection_Freq"])
        values = df[["Infection_Freq"]].values
        from sklearn.preprocessing import StandardScaler

        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(values)
            self.is_fitted = True
        else:
            scaled = self.scaler.transform(values)
        df["Infection_Freq_scaled"] = scaled.flatten()
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Bacteria_encoded" in df.columns:
            df["Bacteria_risk_interaction"] = (
                df["Bacteria_encoded"] * df.get("Total_risk_factors", 0)
            )
        if "Age" in df.columns:
            df["Age_infection_product"] = df["Age"] * df.get("Infection_Freq", 0.0)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude_cols = [
            "ID", "Name", "Email", "Address", "age/gender", "Souches",
            "Collection_Date", "Collection_Date_parsed", "Notes",
            "Bacteria", "Gender", "Age_group", "Days_since_collection",
        ]
        exclude_cols.extend(ANTIBIOTIC_CODES)
        antibiotic_binary_cols = []
        for ab in ANTIBIOTIC_CODES:
            antibiotic_binary_cols.extend([f"{ab}_binary", f"{ab}_S", f"{ab}_R", f"{ab}_I"])
        exclude_cols.extend(antibiotic_binary_cols)
        return [col for col in df.columns if col not in exclude_cols]


class PatientDataAgent:
    """Agent 1: ingest + preprocess patient/bacteria/antibiotic data."""

    def __init__(
        self,
        data_cleaner: Optional[DataCleaner] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ):
        self.data_cleaner = data_cleaner or DataCleaner()
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def ingest(
        self,
        patient_records: RecordType,
        bacteria_metadata: Optional[RecordType] = None,
        antibiotic_panel: Optional[RecordType] = None,
    ) -> pd.DataFrame:
        patient_df = self._ensure_dataframe(patient_records, name="patient_records")

        if bacteria_metadata is not None:
            bacteria_df = self._ensure_dataframe(bacteria_metadata, name="bacteria_metadata")
            patient_df = self._merge_metadata(patient_df, bacteria_df, prefix="Bacteria_")

        if antibiotic_panel is not None:
            antibiotics_df = self._ensure_dataframe(antibiotic_panel, name="antibiotic_panel")
            patient_df = self._merge_metadata(patient_df, antibiotics_df, prefix="LabPanel_")

        return patient_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = self.data_cleaner.clean(df)
        features = self.feature_engineer.engineer_features(cleaned)
        return features

    def prepare(
        self,
        patient_records: RecordType,
        bacteria_metadata: Optional[RecordType] = None,
        antibiotic_panel: Optional[RecordType] = None,
    ) -> pd.DataFrame:
        merged = self.ingest(patient_records, bacteria_metadata, antibiotic_panel)
        processed = self.preprocess(merged)
        return processed

    def prepare_training_dataset(
        self,
        patient_records: RecordType,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        dataset = self._ensure_dataframe(patient_records, name="patient_records")
        processed = self.preprocess(dataset)

        feature_cols = self.feature_engineer.get_feature_columns(processed)
        X = processed[feature_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        y = self._extract_antibiotic_labels(processed)
        return X, y, feature_cols

    def _extract_antibiotic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        label_data = {}
        for code in ANTIBIOTIC_CODES:
            binary_col = f"{code}_binary"
            if binary_col in df.columns:
                label_data[code] = df[binary_col].fillna(0).astype(int)

        if not label_data:
            raise ValueError("Không tìm thấy cột nhãn kháng sinh *_binary trong dữ liệu.")

        return pd.DataFrame(label_data, index=df.index)

    @staticmethod
    def _ensure_dataframe(records: RecordType, name: str) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records.copy()
        if isinstance(records, pd.Series):
            return pd.DataFrame([records.to_dict()])
        if isinstance(records, dict):
            return pd.DataFrame([records])
        if isinstance(records, list):
            return pd.DataFrame(records)
        raise ValueError(f"{name} phải là dict/list/pd.Series/pd.DataFrame")

    @staticmethod
    def _merge_metadata(
        patient_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        if metadata_df.empty:
            return patient_df

        merged = patient_df.copy()
        metadata_row = metadata_df.iloc[0].to_dict()
        for col, value in metadata_row.items():
            new_col = col if col in merged.columns else f"{prefix}{col}"
            if new_col not in merged.columns or merged[new_col].isna().all():
                merged[new_col] = value
        return merged

