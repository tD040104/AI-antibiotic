# Hệ Thống AI Đa Tác Nhân Phân Tích Kháng Kháng Sinh

Hệ thống sử dụng kiến trúc multi-agent để phân tích và dự đoán kháng kháng sinh từ dữ liệu vi khuẩn.

## Kiến Trúc Hệ Thống

Hệ thống bao gồm 6 agents chuyên biệt:

### 1. Agent 1 - Data Cleaner & Normalizer
- Làm sạch dữ liệu đầu vào
- Chuẩn hóa cột ngày tháng, tuổi, giới tính
- Xử lý dữ liệu thiếu
- Chuẩn hóa nhãn S/R/I (Sensitive/Resistant/Intermediate)

### 2. Agent 2 - Feature Engineer
- Mã hóa loại vi khuẩn (one-hot encoding)
- Tạo biến lâm sàng (yếu tố nguy cơ)
- Chuẩn hóa số lần nhiễm trùng
- Tạo đặc trưng tương tác

### 3. Agent 3 - Antibiotic Resistance Predictor
- Mô hình học máy chính (XGBoost/Random Forest)
- Dự đoán xác suất kháng thuốc cho từng loại kháng sinh
- Multi-output classification

### 4. Agent 4 - Treatment Recommender
- Gợi ý thuốc điều trị dựa trên dự đoán
- Áp dụng quy tắc y học (guideline WHO/CLSI)
- Phân loại và ưu tiên kháng sinh

### 5. Agent 5 - Explainability & Report Generator
- Giải thích kết quả bằng SHAP/LIME
- Tạo báo cáo bằng ngôn ngữ tự nhiên
- Xác định yếu tố quan trọng

### 6. Agent 6 - Continuous Learner
- Theo dõi hiệu suất mô hình
- Tự động retrain khi có dữ liệu mới
- Lưu lịch sử hiệu suất

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Sử Dụng

### Huấn luyện hệ thống

```python
from main import MultiAgentOrchestrator

# Khởi tạo orchestrator
orchestrator = MultiAgentOrchestrator(model_type='xgboost')

# Huấn luyện
orchestrator.train('Bacteria_dataset_Multiresictance.csv')
```

### Dự đoán cho bệnh nhân mới

```python
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

### Chạy từ command line

```bash
python main.py
```

## Cấu Trúc Dự Án

```
.
├── agents/
│   ├── agent1_data_cleaner.py
│   ├── agent2_feature_engineer.py
│   ├── agent3_resistance_predictor.py
│   ├── agent4_treatment_recommender.py
│   ├── agent5_explainability.py
│   └── agent6_continuous_learner.py
├── main.py
├── requirements.txt
├── README.md
├── models/              # Mô hình đã lưu
├── logs/                # Lịch sử hiệu suất
└── reports/             # Báo cáo
```

## Dữ Liệu Đầu Vào

File CSV cần có các cột:
- `age/gender`: Tuổi và giới tính
- `Souches`: Tên vi khuẩn
- `Diabetes`, `Hypertension`, `Hospital_before`: Các yếu tố nguy cơ (Yes/No)
- `Infection_Freq`: Tần suất nhiễm trùng
- `AMX/AMP`, `AMC`, `CZ`, `FOX`, `CTX/CRO`, `IPM`, `GEN`, `AN`, ...: Kết quả kháng sinh (S/R/I)
- `Collection_Date`: Ngày thu thập mẫu

## Lưu ý

- Hệ thống tự động xử lý dữ liệu thiếu và chuẩn hóa
- Mô hình được lưu tự động sau khi huấn luyện
- Có thể theo dõi hiệu suất và tự động retrain
- Báo cáo được tạo tự động bằng tiếng Việt

## Tác Giả

Hệ thống được phát triển để phân tích kháng kháng sinh trong y học lâm sàng.




