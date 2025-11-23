"""
Main Orchestrator - Điều phối các agents trong hệ thống multi-agent
Updated to use new src/ structure
"""

import pandas as pd
import numpy as np
import os
import joblib

# Try new structure first, fallback to old
try:
    from src.preprocessing import DataCleaner, FeatureEngineer
    from src.modeling import ResistancePredictor
    NEW_STRUCTURE = True
except ImportError:
    from agents.agent1_data_cleaner import DataCleanerAgent as DataCleaner
    from agents.agent2_feature_engineer import FeatureEngineerAgent as FeatureEngineer
    from agents.agent3_resistance_predictor import ResistancePredictorAgent as ResistancePredictor
    NEW_STRUCTURE = False

try:
    from agents.agent4_treatment_recommender import TreatmentRecommenderAgent
    from agents.agent5_explainability import ExplainabilityAgent
    from agents.agent6_continuous_learner import ContinuousLearnerAgent
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False


class MultiAgentOrchestrator:
    def __init__(self, model_type: str = 'xgboost'):
        """
        Khởi tạo orchestrator với các agents
        """
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.model = ResistancePredictor(model_type=model_type)
        
        if HAS_AGENTS:
            self.agent4 = TreatmentRecommenderAgent()
            self.agent5 = ExplainabilityAgent()
            self.agent6 = ContinuousLearnerAgent()
        
        # For backward compatibility
        self.agent1 = self.data_cleaner
        self.agent2 = self.feature_engineer
        self.agent3 = self.model
        
        self.is_trained = False
        self.feature_columns = None
        self.X_columns = None
        
    def train(self, csv_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Huấn luyện toàn bộ hệ thống từ file CSV
        """
        print("=" * 80)
        print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN")
        print("=" * 80)
        
        # Bước 1: Load và làm sạch dữ liệu
        print("\n[Bước 1] Đang làm sạch dữ liệu...")
        df = pd.read_csv(csv_path)
        
        # Tách age/gender nếu tồn tại
        if 'age/gender' in df.columns:
            extracted = df['age/gender'].astype(str).str.extract(r'(?P<age>\d+)\s*/\s*(?P<gender>[MF])')
            df['age'] = pd.to_numeric(extracted['age'], errors='coerce')
            df['gender'] = extracted['gender']
        
        df_cleaned = self.data_cleaner.clean(df)
        print(f"  ✓ Dữ liệu sau khi làm sạch: {len(df_cleaned)} mẫu")
        
        # Bước 2: Feature engineering
        print("\n[Bước 2] Đang tạo đặc trưng...")
        df_features = self.feature_engineer.engineer_features(df_cleaned)
        self.feature_columns = self.feature_engineer.get_feature_columns(df_features)
        print(f"  ✓ Số đặc trưng: {len(self.feature_columns)}")
        
        # Bước 3: Chuẩn bị dữ liệu training
        print("\n[Bước 3] Đang chuẩn bị dữ liệu training...")
        
        # Lấy các cột đặc trưng
        X = df_features[self.feature_columns].copy()
        
        # Loại bỏ các cột không phải numeric
        numeric_cols = []
        for col in X.columns:
            dtype = X[col].dtype
            if pd.api.types.is_numeric_dtype(dtype) or dtype == 'bool':
                numeric_cols.append(col)
            else:
                print(f"  ⚠ Loại bỏ cột non-numeric: {col} (dtype: {dtype})")
        
        X = X[numeric_cols].copy()
        self.feature_columns = numeric_cols
        self.X_columns = X.columns.tolist()
        
        # Lấy nhãn (binary encoding cho các kháng sinh)
        antibiotic_cols = [
            'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
            'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole',
            'Furanes', 'colistine'
        ]
        
        y_columns = [f'{col}_binary' for col in antibiotic_cols]
        y = df_features[y_columns].copy()
        y.columns = antibiotic_cols
        
        # Loại bỏ các hàng có quá nhiều giá trị thiếu
        valid_idx = y.notna().sum(axis=1) >= len(antibiotic_cols) * 0.5
        X = X[valid_idx].copy()
        y = y[valid_idx].copy()
        
        # Điền NaN bằng 0
        y = y.fillna(0).astype(int)
        X = X.fillna(0)
        
        # Chuyển đổi tất cả các cột về numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"  ✓ Số mẫu training: {len(X)}")
        
        # Bước 4: Huấn luyện mô hình
        print("\n[Bước 4] Đang huấn luyện mô hình...")
        results = self.model.train(X, y, test_size=test_size, random_state=random_state)
        
        print(f"\n  ✓ Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"  ✓ Test Accuracy: {results['test_accuracy']:.4f}")
        
        # Bước 5: Lưu mô hình
        os.makedirs("models", exist_ok=True)
        self.model.save_model("models/model_latest.pkl")
        
        # Bước 6: Monitor performance (nếu có agent6)
        if HAS_AGENTS:
            print("\n[Bước 6] Đang theo dõi hiệu suất...")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            _, y_proba = self.model.predict(X_test)
            performance = self.agent6.monitor_performance(
                self.model, X_test, y_test, "initial_training"
            )
            print(f"  ✓ Performance logged: Accuracy = {performance['accuracy']:.4f}")
        
        self.is_trained = True
        
        # Lưu trạng thái phục vụ suy luận
        self.save_state("models/orchestrator_state.joblib")
        
        print("\n" + "=" * 80)
        print("HUẤN LUYỆN HOÀN TẤT!")
        print("=" * 80)
        
        return results
    
    def save_state(self, filepath: str):
        """Lưu toàn bộ trạng thái cần thiết để suy luận"""
        if not self.is_trained:
            raise ValueError("Không thể lưu trạng thái khi mô hình chưa được huấn luyện.")
        
        state = {
            'model_type': self.model.model_type if hasattr(self.model, 'model_type') else 'xgboost',
            'data_cleaner': self.data_cleaner,
            'feature_engineer': self.feature_engineer,
            'model': self.model,
            'feature_columns': self.feature_columns,
            'X_columns': self.X_columns,
        }
        
        if HAS_AGENTS:
            state['agent4'] = self.agent4
            state['agent5'] = self.agent5
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump(state, filepath)
        print(f"Trạng thái hệ thống đã được lưu tại: {filepath}")
    
    @classmethod
    def load_from_state(cls, filepath: str):
        """Khởi tạo orchestrator từ trạng thái đã lưu"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file trạng thái: {filepath}")
        
        state = joblib.load(filepath)
        model_type = state.get('model_type', 'xgboost')
        orchestrator = cls(model_type=model_type)
        
        orchestrator.data_cleaner = state.get('data_cleaner', orchestrator.data_cleaner)
        orchestrator.feature_engineer = state.get('feature_engineer', orchestrator.feature_engineer)
        orchestrator.model = state.get('model', orchestrator.model)
        
        if HAS_AGENTS:
            orchestrator.agent4 = state.get('agent4', TreatmentRecommenderAgent())
            orchestrator.agent5 = state.get('agent5', ExplainabilityAgent())
        
        orchestrator.feature_columns = state.get('feature_columns')
        orchestrator.X_columns = state.get('X_columns')
        orchestrator.is_trained = True
        orchestrator.model.is_trained = True
        
        # Backward compatibility
        orchestrator.agent1 = orchestrator.data_cleaner
        orchestrator.agent2 = orchestrator.feature_engineer
        orchestrator.agent3 = orchestrator.model
        
        return orchestrator
    
    def predict(self, patient_data: dict) -> dict:
        """Dự đoán kháng thuốc cho một bệnh nhân mới"""
        if not self.is_trained:
            raise ValueError("Hệ thống chưa được huấn luyện! Vui lòng gọi train() trước.")
        
        try:
            # Chuyển đổi patient_data thành DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Tách age/gender nếu có
            if 'age/gender' in patient_df.columns:
                extracted = patient_df['age/gender'].astype(str).str.extract(r'(?P<age>\d+)\s*/\s*(?P<gender>[MF])')
                patient_df['age'] = pd.to_numeric(extracted['age'], errors='coerce')
                patient_df['gender'] = extracted['gender']
            
            # Data cleaning
            patient_cleaned = self.data_cleaner.clean(patient_df)
            
            # Feature engineering
            patient_features = self.feature_engineer.engineer_features(patient_cleaned)
            
            # Đảm bảo có đủ các cột đặc trưng
            for col in self.X_columns:
                if col not in patient_features.columns:
                    patient_features[col] = 0
            
            # Chọn đúng các cột đặc trưng
            X_patient = patient_features[self.X_columns].fillna(0)
            
            # Đảm bảo tất cả các cột đều là numeric
            X_patient = X_patient.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Predict
            predictions, probabilities = self.model.predict(X_patient)
            
            # Treatment recommendations (if available)
            recommendations = []
            if HAS_AGENTS:
                patient_series = patient_features.iloc[0]
                recommendations = self.agent4.recommend_treatment(
                    probabilities, 
                    patient_series,
                    top_k=5
                )
                
                # Explanation
                explanation = self.agent5.explain_prediction(
                    patient_series,
                    predictions,
                    probabilities
                )
                
                # Detailed report
                detailed_report = self.agent5.generate_detailed_report(explanation)
            else:
                explanation = {}
                detailed_report = "Báo cáo chi tiết không khả dụng (thiếu agent5)"
            
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


def main():
    """Hàm main để chạy hệ thống"""
    orchestrator = MultiAgentOrchestrator(model_type='xgboost')
    
    # Huấn luyện hệ thống
    # Try data/ first, then root
    csv_path = "data/Bacteria_dataset_Multiresictance.csv"
    if not os.path.exists(csv_path):
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
            'Souches': 'S999',
            'AMX/AMP_binary': 1,
            'AMC_binary': 0,
            'CZ_binary': 0,
            'FOX_binary': 0,
            'CTX/CRO_binary': 1,
            'IPM_binary': 0,
            'GEN_binary': 0,
            'AN_binary': 0,
            'Acide nalidixique_binary': 0,
            'ofx_binary': 0,
            'CIP_binary': 1,
            'C_binary': 0,
            'Co-trimoxazole_binary': 0,
            'Furanes_binary': 0,
            'colistine_binary': 0
        }
        
        try:
            result = orchestrator.predict(sample_patient)
            
            print("\nKẾT QUẢ DỰ ĐOÁN:")
            for antibiotic, prediction in result['predictions'].items():
                status = "Nhạy" if prediction == 1 else "Kháng"
                print(f"  - {antibiotic}: {status}")
            
            print("\nXÁC SUẤT DỰ ĐOÁN:")
            for antibiotic, probability in result['probabilities'].items():
                print(f"  - {antibiotic}: {probability:.1%}")
            
            # Report
            print("\nBÁO CÁO CHI TIẾT:")
            if 'report' in result and result['report']:
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
        print("Vui lòng đảm bảo file dữ liệu nằm trong thư mục data/")


if __name__ == "__main__":
    main()
