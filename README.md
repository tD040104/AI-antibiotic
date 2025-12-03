# Hệ Thống AI Đa Tác Nhân Phân Tích Kháng Kháng Sinh

Hệ thống sử dụng kiến trúc multi-agent để phân tích và dự đoán kháng kháng sinh từ dữ liệu vi khuẩn.

## Cấu Trúc Dự Án

```
.
├── agents/                  # Bộ 5 agent MAS
│   ├── patient_data_agent.py
│   ├── prediction_agent.py
│   ├── explainability_agent.py
│   ├── critic_agent.py
│   └── decision_agent.py
├── data/                    # Dữ liệu huấn luyện
│   └── Bacteria_dataset_Multiresictance.csv
├── demo/                    # Ứng dụng Streamlit (MAS)
│   └── app.py
├── example_usage.py         # Ví dụ train/predict/batch
├── logs/                    # Performance logs
├── main.py                  # MASClinicalDecisionSystem
├── models/                  # Trained model & state (mas_model / mas_state)
├── requirements.txt
└── README.md
```

## Kiến Trúc Hệ Thống

Hệ thống MAS gồm 5 agent lõi được điều phối bởi `MASClinicalDecisionSystem`:

1. **PatientDataAgent**  
   - Nhập dữ liệu bệnh nhân + metadata vi khuẩn/kháng sinh  
   - Làm sạch + tạo đặc trưng (tích hợp DataCleaner & FeatureEngineer)

2. **AntibioticPredictionAgent**  
   - Bao bọc mô hình multi-output (XGBoost/RandomForest/GBM)  
   - Huấn luyện, lưu/khôi phục và dự đoán xác suất nhạy/kháng

3. **ExplainabilityEvaluationAgent**  
   - Tạo báo cáo SHAP/LIME (nếu có) + giải thích tự nhiên  
   - Tổng hợp yếu tố quan trọng ảnh hưởng kết quả

4. **CriticAgent**  
   - Gắn cờ ca bệnh có xác suất ~0.5 hoặc thiếu dữ liệu  
   - Đề xuất yêu cầu xét nghiệm/điền bổ sung thông tin

5. **DecisionAgent**  
   - Kết hợp critic + xác suất nhạy để đề xuất kháng sinh  
   - Xuất hành động chính (dùng thuốc, yêu cầu xét nghiệm, v.v.)

`ClinicalDecisionPipeline` kết nối các agent trên cho cả huấn luyện và suy luận (được demo trong `example_usage.py` và `demo/app.py`).

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Sử Dụng

### 1. Huấn luyện hệ thống (MAS 5-Agent)

```python
from main import MASClinicalDecisionSystem

system = MASClinicalDecisionSystem(model_type="xgboost")
system.train("data/Bacteria_dataset_Multiresictance.csv")
```

Sau khi huấn luyện, mô hình và trạng thái được lưu tại `models/mas_model.pkl`
và `models/mas_state.joblib`.

### 2. Tải lại mô hình MAS để suy luận

```python
from main import MASClinicalDecisionSystem

system = MASClinicalDecisionSystem()
system.load(
    model_path="models/mas_model.pkl",
    state_path="models/mas_state.joblib",
)

patient = {
    "age/gender": "45/F",
    "Souches": "Escherichia coli",
    "Diabetes": "Yes",
    "Hypertension": "No",
    "Hospital_before": "Yes",
    "Infection_Freq": 2.0,
    "Collection_Date": "2024-01-15",
}

result = system.predict(patient)
print(result["decision"]["primary_actions"])
```

### 3. Chạy ví dụ CLI

```bash
python example_usage.py
```

File ví dụ sẽ:
- Huấn luyện hệ thống (hoặc bạn có thể sửa thành `system.load(...)`)
- Dự đoán cho 1 bệnh nhân mẫu
- Chạy batch 5 bệnh nhân đầu tiên trong CSV

### 4. Chạy Demo Application (Streamlit)

```bash
streamlit run demo/app.py
```

Ứng dụng sẽ yêu cầu đường dẫn `models/mas_model.pkl` và `models/mas_state.joblib`, sau đó cho phép nhập thông tin bệnh nhân và xem:
- Nhãn nhạy/kháng + xác suất
- Cảnh báo từ Critic Agent
- Hành động ưu tiên + khuyến nghị kháng sinh

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
- Toàn bộ pipeline MAS nằm trong `agents/` và `main.py`

## Tác Giả

Hệ thống được phát triển để phân tích kháng kháng sinh trong y học lâm sàng.
