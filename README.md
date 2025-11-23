# Hệ Thống AI Đa Tác Nhân Phân Tích Kháng Kháng Sinh

Hệ thống sử dụng kiến trúc multi-agent để phân tích và dự đoán kháng kháng sinh từ dữ liệu vi khuẩn.

## Cấu Trúc Dự Án

```
.
├── data/                    # Dữ liệu (không upload raw large data nếu có)
│   └── Bacteria_dataset_Multiresictance.csv
├── notebooks/               # Jupyter notebooks
│   └── notebook.ipynb
├── src/                     # Source code
│   ├── preprocessing.py     # Data cleaning và feature engineering
│   ├── modeling.py          # Machine learning model
│   └── predict.py           # Prediction pipeline
├── demo/                    # Demo application
│   └── app.py              # Streamlit app
├── models/                  # Trained models
│   ├── model_latest.pkl
│   └── orchestrator_state.joblib
├── agents/                  # Legacy agents (backward compatibility)
│   ├── agent1_data_cleaner.py
│   ├── agent2_feature_engineer.py
│   ├── agent3_resistance_predictor.py
│   ├── agent4_treatment_recommender.py
│   ├── agent5_explainability.py
│   ├── agent6_continuous_learner.py
│   └── agent7_clinical_decision.py
├── logs/                    # Performance logs
├── main.py                  # Main orchestrator
├── requirements.txt
└── README.md
```

## Kiến Trúc Hệ Thống

Hệ thống được tổ chức lại theo cấu trúc module:

### 1. Preprocessing (`src/preprocessing.py`)
- **DataCleaner**: Làm sạch dữ liệu đầu vào, chuẩn hóa các cột, xử lý dữ liệu thiếu
- **FeatureEngineer**: Mã hóa loại vi khuẩn, tạo biến lâm sàng, tạo đặc trưng tương tác

### 2. Modeling (`src/modeling.py`)
- **ResistancePredictor**: Mô hình học máy chính (XGBoost/Random Forest) để dự đoán kháng thuốc

### 3. Prediction (`src/predict.py`)
- **Predictor**: Pipeline dự đoán hoàn chỉnh, xử lý từ input đến output

### 4. Demo Application (`demo/app.py`)
- **Streamlit App**: Ứng dụng web với:
  - Input: Đặc trưng bệnh nhân (tuổi/giới tính, vi khuẩn, yếu tố nguy cơ, v.v.)
  - Output: Thông tin kháng/nhạy cho từng kháng sinh

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Sử Dụng

### 1. Huấn luyện hệ thống

```python
from main import MultiAgentOrchestrator

# Khởi tạo orchestrator
orchestrator = MultiAgentOrchestrator(model_type='xgboost')

# Huấn luyện
orchestrator.train('data/Bacteria_dataset_Multiresictance.csv')
```

Hoặc chạy từ command line:

```bash
python main.py
```

### 2. Sử dụng Predictor (cấu trúc mới)

```python
from src.predict import Predictor

# Load model
predictor = Predictor()
predictor.load_model(
    model_path="models/model_latest.pkl",
    state_path="models/orchestrator_state.joblib"
)

# Dự đoán
patient_data = {
    'age/gender': '45/F',
    'Souches': 'Escherichia coli',
    'Diabetes': 'Yes',
    'Hypertension': 'No',
    'Hospital_before': 'Yes',
    'Infection_Freq': 2.0,
    'Collection_Date': '2024-01-15'
}

result = predictor.predict(patient_data)

# Xem kết quả
print("Kháng sinh nhạy:", result['resistance_info']['sensitive_count'])
print("Kháng sinh kháng:", result['resistance_info']['resistant_count'])
```

### 3. Chạy Demo Application (Streamlit)

```bash
streamlit run demo/app.py
```

Ứng dụng sẽ mở trong trình duyệt với:
- Form nhập thông tin bệnh nhân
- Hiển thị kết quả dự đoán kháng/nhạy
- Biểu đồ và bảng chi tiết

### 4. Sử dụng Orchestrator (backward compatibility)

```python
from main import MultiAgentOrchestrator

# Load từ state đã lưu
orchestrator = MultiAgentOrchestrator.load_from_state("models/orchestrator_state.joblib")

# Dự đoán
patient_data = {
    'age/gender': '45/F',
    'Souches': 'Escherichia coli',
    'Diabetes': 'Yes',
    'Hypertension': 'No',
    'Hospital_before': 'Yes',
    'Infection_Freq': 2.0,
    'Collection_Date': '2024-01-15'
}

result = orchestrator.predict(patient_data)

# In báo cáo
print(result['report'])

# Xem khuyến nghị
for rec in result['recommendations']:
    print(f"{rec['rank']}. {rec['antibiotic_name']}")
    print(f"   Xác suất nhạy: {rec['sensitive_probability']:.1%}")
```

## Dữ Liệu Đầu Vào

File CSV cần có các cột:
- `age/gender`: Tuổi và giới tính (ví dụ: "45/F", "68/M")
- `Souches`: Tên vi khuẩn
- `Diabetes`, `Hypertension`, `Hospital_before`: Các yếu tố nguy cơ (Yes/No)
- `Infection_Freq`: Tần suất nhiễm trùng
- `AMX/AMP`, `AMC`, `CZ`, `FOX`, `CTX/CRO`, `IPM`, `GEN`, `AN`, ...: Kết quả kháng sinh (S/R/I)
- `Collection_Date`: Ngày thu thập mẫu

## Demo Application

Ứng dụng Streamlit (`demo/app.py`) cung cấp giao diện web để:

1. **Tải mô hình**: Chọn đường dẫn đến model đã huấn luyện
2. **Nhập thông tin bệnh nhân**:
   - Tuổi/Giới tính
   - Tên vi khuẩn
   - Các yếu tố nguy cơ (tiểu đường, tăng huyết áp, tiền sử nhập viện)
   - Tần suất nhiễm trùng
   - Ngày thu thập mẫu

3. **Xem kết quả**:
   - Số lượng kháng sinh nhạy/kháng
   - Bảng chi tiết kháng sinh nhạy
   - Bảng chi tiết kháng sinh kháng
   - Biểu đồ xác suất
   - Bảng xác suất đầy đủ

## Lưu ý

- Hệ thống tự động xử lý dữ liệu thiếu và chuẩn hóa
- Mô hình được lưu tự động sau khi huấn luyện
- Có thể theo dõi hiệu suất và tự động retrain
- Báo cáo được tạo tự động bằng tiếng Việt
- Cấu trúc mới trong `src/` tương thích với cấu trúc cũ trong `agents/`

## Tác Giả

Hệ thống được phát triển để phân tích kháng kháng sinh trong y học lâm sàng.
