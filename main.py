"""
Main Orchestrator - Điều phối các agents trong hệ thống multi-agent
"""

import pandas as pd
import numpy as np
from agents.agent1_data_cleaner import DataCleanerAgent
from agents.agent2_feature_engineer import FeatureEngineerAgent
from agents.agent3_resistance_predictor import ResistancePredictorAgent
from agents.agent4_treatment_recommender import TreatmentRecommenderAgent
from agents.agent5_explainability import ExplainabilityAgent
from agents.agent6_continuous_learner import ContinuousLearnerAgent
import os
import joblib


class MultiAgentOrchestrator:
    def __init__(self, model_type: str = 'xgboost'):
        """
        Khởi tạo orchestrator với các agents
        """
        self.agent1 = DataCleanerAgent()
        self.agent2 = FeatureEngineerAgent()
        self.agent3 = ResistancePredictorAgent(model_type=model_type)
        self.agent4 = TreatmentRecommenderAgent()
        self.agent5 = ExplainabilityAgent()
        self.agent6 = ContinuousLearnerAgent()
        
        self.is_trained = False
        self.feature_columns = None
        
    def train(self, csv_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Huấn luyện toàn bộ hệ thống từ file CSVY
        """
        print("=" * 80)
        print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN")
        print("=" * 80)
        
        # Bước 1: Load và làm sạch dữ liệu (Agent 1)
        print("\n[Agent 1] Đang làm sạch dữ liệu...")
        df = pd.read_csv(csv_path)
        df_cleaned = self.agent1.clean(df)
        print(f"  ✓ Dữ liệu sau khi làm sạch: {len(df_cleaned)} mẫu")
        
        # Bước 2: Feature engineering (Agent 2)
        print("\n[Agent 2] Đang tạo đặc trưng...")
        df_features = self.agent2.engineer_features(df_cleaned)
        self.feature_columns = self.agent2.get_feature_columns(df_features)
        print(f"  ✓ Số đặc trưng: {len(self.feature_columns)}")
        
        # Bước 3: Chuẩn bị dữ liệu training
        print("\n[Agent 3] Đang chuẩn bị dữ liệu training...")
        
        # Lấy các cột đặc trưng
        X = df_features[self.feature_columns].copy()
        
        # Loại bỏ các cột không phải numeric (object, datetime, etc.)
        # XGBoost chỉ chấp nhận int, float, bool, category
        numeric_cols = []
        for col in X.columns:
            dtype = X[col].dtype
            if pd.api.types.is_numeric_dtype(dtype) or dtype == 'bool':
                numeric_cols.append(col)
            else:
                print(f"  ⚠ Loại bỏ cột non-numeric: {col} (dtype: {dtype})")
        
        X = X[numeric_cols].copy()
        self.feature_columns = numeric_cols  # Cập nhật lại feature columns
        
        # Lấy nhãn (binary encoding cho các kháng sinh)
        antibiotic_cols = [
            'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
            'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole',
            'Furanes', 'colistine'
        ]
        
        y_columns = [f'{col}_binary' for col in antibiotic_cols]
        y = df_features[y_columns].copy()
        y.columns = antibiotic_cols  # Đổi tên cột về tên gốc
        
        # Loại bỏ các hàng có quá nhiều giá trị thiếu
        valid_idx = y.notna().sum(axis=1) >= len(antibiotic_cols) * 0.5
        X = X[valid_idx].copy()
        y = y[valid_idx].copy()
        
        # Điền NaN bằng 0 (giả định là Resistant)
        y = y.fillna(0).astype(int)
        X = X.fillna(0)
        
        # Chuyển đổi tất cả các cột về numeric (đảm bảo không có object)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"  ✓ Số mẫu training: {len(X)}")
        
        # Bước 4: Huấn luyện mô hình (Agent 3)
        print("\n[Agent 3] Đang huấn luyện mô hình...")
        results = self.agent3.train(X, y, test_size=test_size, random_state=random_state)
        
        print(f"\n  ✓ Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"  ✓ Test Accuracy: {results['test_accuracy']:.4f}")
        
        # Bước 5: Lưu mô hình
        os.makedirs("models", exist_ok=True)
        self.agent3.save_model("models/model_latest.pkl")
        
        # Bước 6: Monitor performance (Agent 6)
        print("\n[Agent 6] Đang theo dõi hiệu suất...")
        # Tách test set để monitor
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        _, y_proba = self.agent3.predict(X_test)
        # Tạo y_test binary từ probabilities để monitor
        performance = self.agent6.monitor_performance(
            self.agent3, X_test, y_test, "initial_training"
        )
        print(f"  ✓ Performance logged: Accuracy = {performance['accuracy']:.4f}")
        
        self.is_trained = True
        self.X_columns = X.columns.tolist()
        
        # Lưu trạng thái phục vụ suy luận
        self.save_state("models/orchestrator_state.joblib")
        
        print("\n" + "=" * 80)
        print("HUẤN LUYỆN HOÀN TẤT!")
        print("=" * 80)
        
        return results
    
    def save_state(self, filepath: str):
        """
        Lưu toàn bộ trạng thái cần thiết để suy luận
        """
        if not self.is_trained:
            raise ValueError("Không thể lưu trạng thái khi mô hình chưa được huấn luyện.")
        
        state = {
            'model_type': self.agent3.model_type if hasattr(self.agent3, 'model_type') else 'xgboost',
            'agent1': self.agent1,
            'agent2': self.agent2,
            'agent3': self.agent3,
            'agent4': self.agent4,
            'agent5': self.agent5,
            'feature_columns': self.feature_columns,
            'X_columns': self.X_columns,
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump(state, filepath)
        print(f"Trạng thái hệ thống đã được lưu tại: {filepath}")
    
    @classmethod
    def load_from_state(cls, filepath: str):
        """
        Khởi tạo orchestrator từ trạng thái đã lưu
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file trạng thái: {filepath}")
        
        state = joblib.load(filepath)
        orchestrator = cls(model_type=state.get('model_type', 'xgboost'))
        
        orchestrator.agent1 = state['agent1']
        orchestrator.agent2 = state['agent2']
        orchestrator.agent3 = state['agent3']
        orchestrator.agent4 = state.get('agent4', TreatmentRecommenderAgent())
        orchestrator.agent5 = state.get('agent5', ExplainabilityAgent())
        
        orchestrator.feature_columns = state.get('feature_columns')
        orchestrator.X_columns = state.get('X_columns')
        orchestrator.is_trained = True
        orchestrator.agent3.is_trained = True
        
        return orchestrator
    
    def predict(self, patient_data: dict) -> dict:
        """
        Dự đoán kháng thuốc cho một bệnh nhân mới
        """
        if not self.is_trained:
            raise ValueError("Hệ thống chưa được huấn luyện! Vui lòng gọi train() trước.")
        
        try:
            # Chuyển đổi patient_data thành DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Agent 1: Làm sạch
            patient_cleaned = self.agent1.clean(patient_df)
            
            # Agent 2: Feature engineering
            patient_features = self.agent2.engineer_features(patient_cleaned)
            
            # Đảm bảo có đủ các cột đặc trưng
            for col in self.X_columns:
                if col not in patient_features.columns:
                    patient_features[col] = 0
            
            # Chọn đúng các cột đặc trưng
            X_patient = patient_features[self.X_columns].fillna(0)
            
            # Đảm bảo tất cả các cột đều là numeric
            X_patient = X_patient.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Kiểm tra số lượng features
            if len(X_patient.columns) != len(self.X_columns):
                missing = set(self.X_columns) - set(X_patient.columns)
                extra = set(X_patient.columns) - set(self.X_columns)
                if missing:
                    print(f"  ⚠ Thiếu {len(missing)} features: {list(missing)[:5]}...")
                if extra:
                    print(f"  ⚠ Thừa {len(extra)} features: {list(extra)[:5]}...")
            
            # Agent 3: Dự đoán
            predictions, probabilities = self.agent3.predict(X_patient)
            
            # Agent 4: Gợi ý điều trị
            patient_series = patient_features.iloc[0]
            recommendations = self.agent4.recommend_treatment(
                probabilities, 
                patient_series,
                top_k=5
            )
            
            # Agent 5: Giải thích và báo cáo
            explanation = self.agent5.explain_prediction(
                patient_series,
                predictions,
                probabilities
            )
            
            # Tạo báo cáo chi tiết
            detailed_report = self.agent5.generate_detailed_report(explanation)
            
            return {
                'predictions': predictions.iloc[0].to_dict(),
                'probabilities': probabilities.iloc[0].to_dict(),
                'recommendations': recommendations,
                'explanation': explanation,
                'report': detailed_report
            }
        except Exception as e:
            print(f"  ❌ Lỗi trong quá trình predict: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def update_with_new_data(self, new_csv_path: str, retrain: bool = True):
        """
        Cập nhật mô hình với dữ liệu mới
        """
        print("\n[Agent 6] Nhận dữ liệu mới, đang cập nhật...")
        
        if retrain:
            # Retrain với toàn bộ dữ liệu
            # Load dữ liệu cũ và mới
            # (Trong thực tế, cần lưu trữ dữ liệu training ban đầu)
            print("  → Cần retrain với toàn bộ dữ liệu")
            # self.train(new_csv_path)  # Uncomment nếu có dữ liệu đầy đủ
        else:
            # Incremental update
            new_df = pd.read_csv(new_csv_path)
            # Process và update
            pass


def main():
    """
    Hàm main để chạy hệ thống
    """
    # Khởi tạo orchestrator
    orchestrator = MultiAgentOrchestrator(model_type='xgboost')
    
    # Huấn luyện hệ thống
    csv_path = "Bacteria_dataset_Multiresictance.csv"
    
    if os.path.exists(csv_path):
        print("Bắt đầu huấn luyện hệ thống...\n")
        results = orchestrator.train(csv_path, test_size=0.2)
        
        # Ví dụ dự đoán cho bệnh nhân mới
        print("\n" + "=" * 80)
        print("VÍ DỤ DỰ ĐOÁN CHO BỆNH NHÂN MỚI")
        print("=" * 80)
        
        sample_patient = {
            'age/gender': '45/F',
            'Souches': 'S999 Escherichia coli',
            'Diabetes': 'Yes',
            'Hypertension': 'No',
            'Hospital_before': 'Yes',
            'Infection_Freq': 2.0,
            'Collection_Date': '2024-01-15'
        }
        
        try:
            print("\n[Debug] Đang thực hiện dự đoán...")
            result = orchestrator.predict(sample_patient)
            
            print("\nBÁO CÁO:")
            if result and 'report' in result:
                print(result['report'])
            else:
                print("  ⚠ Không thể tạo báo cáo. Kết quả:", result.keys() if result else "None")
            
            print("\nKHUYẾN NGHỊ ĐIỀU TRỊ:")
            if result and 'recommendations' in result and result['recommendations']:
                for rec in result['recommendations']:
                    print(f"  {rec['rank']}. {rec['antibiotic_name']}")
                    print(f"     Xác suất nhạy: {rec['sensitive_probability']:.1%}")
                    print(f"     Độ tin cậy: {rec['confidence']}")
                    print()
            else:
                print("  ⚠ Không có khuyến nghị điều trị.")
        except Exception as e:
            print(f"\n❌ Lỗi khi dự đoán: {e}")
            print("\nChi tiết lỗi:")
            import traceback
            traceback.print_exc()
    else:
        print(f"Không tìm thấy file: {csv_path}")


if __name__ == "__main__":
    main()

