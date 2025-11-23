"""
Streamlit Demo Application
Input: Patient features
Output: Resistance/Sensitivity information
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import Predictor

# Page config
st.set_page_config(
    page_title="Dự Đoán Kháng Kháng Sinh",
    page_icon="🦠",
    layout="wide"
)

# Title
st.title("🦠 Hệ Thống Dự Đoán Kháng Kháng Sinh")
st.markdown("---")

# CSS để căn giữa nội dung trong các bảng
st.markdown("""
<style>
    div[data-testid="stDataFrame"] table th,
    div[data-testid="stDataFrame"] table td {
        text-align: center !important;
    }
    .dataframe th,
    .dataframe td {
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] table td[data-testid="stDataFrameCell"],
    div[data-testid="stDataFrame"] table td {
        text-align: center !important;
    }
    [data-testid="stExpander"] div[data-testid="stDataFrame"] table th,
    [data-testid="stExpander"] div[data-testid="stDataFrame"] table td {
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] table * {
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] table td[style*="text-align"],
    div[data-testid="stDataFrame"] table td {
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] table td[style] {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

# Common bacteria list
COMMON_BACTERIA = [
    "Escherichia coli", "Klebsiella pneumoniae", "Klebsiella oxytoca",
    "Proteus mirabilis", "Proteus vulgaris", "Enterobacter cloacae",
    "Enterobacter aerogenes", "Serratia marcescens", "Citrobacter freundii",
    "Citrobacter koseri", "Morganella morganii", "Providencia stuartii",
    "Acinetobacter baumannii", "Pseudomonas aeruginosa", "Staphylococcus aureus",
    "Staphylococcus epidermidis", "Enterococcus faecalis", "Enterococcus faecium",
    "Streptococcus pneumoniae", "Streptococcus pyogenes", "Haemophilus influenzae",
    "Neisseria meningitidis", "Salmonella enterica", "Shigella sonnei",
    "Shigella flexneri", "Campylobacter jejuni", "Helicobacter pylori",
    "Bacteroides fragilis", "Clostridium difficile", "Listeria monocytogenes"
]

# Column name mapping → Vietnamese
COLUMN_MAP = {
    'name': 'Tên kháng sinh',
    'code': 'Mã',
    'sensitivity_probability': 'Xác suất nhạy',
    'resistance_probability': 'Xác suất kháng',
    'status': 'Trạng thái'
}

def render_centered_table(df: pd.DataFrame):
    """Render DataFrame as HTML table with centered content"""
    html = "<div style='overflow-x: auto;'>"
    html += "<table style='width: 100%; border-collapse: collapse; margin: 0 auto;'>"
    
    # Header
    html += "<thead><tr>"
    for col in df.columns:
        html += f"<th style='text-align: center; padding: 10px; border: 1px solid #ddd; font-weight: bold;'>{col}</th>"
    html += "</tr></thead>"
    
    # Body
    html += "<tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            value = row[col]
            html += f"<td style='text-align: center; padding: 10px; border: 1px solid #ddd;'>{value}</td>"
        html += "</tr>"
    html += "</tbody>"
    
    html += "</table></div>"
    return html

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài Đặt")
    
    model_path = st.text_input("Đường dẫn mô hình", value="models/model_latest.pkl")
    state_path = st.text_input("Đường dẫn trạng thái", value="models/orchestrator_state.joblib")
    
    if st.button("📥 Tải Mô Hình", type="primary"):
        try:
            predictor = Predictor()
            predictor.load_model(model_path, state_path)
            st.session_state.predictor = predictor
            st.session_state.model_loaded = True
            st.success("✅ Đã tải mô hình thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
            st.session_state.model_loaded = False
    
    if st.session_state.model_loaded:
        st.success("✅ Mô hình đã sẵn sàng")
    else:
        st.warning("⚠️ Vui lòng tải mô hình trước khi dự đoán")

# Main
st.header("📋 Nhập Thông Tin Bệnh Nhân")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Tuổi", 0, 120, 45)
        gender = st.selectbox("Giới tính", ["F", "M"], format_func=lambda x: "Nữ" if x == "F" else "Nam")
        bacteria = st.selectbox("Tên vi khuẩn (Souches)", COMMON_BACTERIA, index=0)
        diabetes = st.selectbox("Tiểu đường", ["No", "Yes"])
    
    with col2:
        hospital_before = st.selectbox("Tiền sử nhập viện", ["No", "Yes"])
        infection_freq = st.number_input("Tần suất nhiễm trùng", 0.0, 10.0, 1.0, step=0.1)
        collection_date = st.date_input("Ngày thu thập mẫu", value=pd.Timestamp.now().date())
        hypertension = st.selectbox("Tăng huyết áp", ["No", "Yes"])
    
    submitted = st.form_submit_button("🔍 Dự Đoán", type="primary", use_container_width=True)

# Prediction
if submitted:
    if not st.session_state.model_loaded:
        st.error("❌ Vui lòng tải mô hình trước!")
    else:
        try:
            patient_data = {
                'age/gender': f"{age}/{gender}",
                'Souches': bacteria,
                'Diabetes': diabetes,
                'Hypertension': hypertension,
                'Hospital_before': hospital_before,
                'Infection_Freq': float(infection_freq),
                'Collection_Date': str(collection_date)
            }
            
            with st.spinner("Đang dự đoán..."):
                result = st.session_state.predictor.predict(patient_data)
            
            st.success("✅ Hoàn tất dự đoán!")
            st.markdown("---")
            
            resistance_info = result['resistance_info']
            
            st.header("📊 Kết Quả Dự Đoán")
            c1, c2 = st.columns(2)
            c1.metric("Kháng sinh nhạy", resistance_info['sensitive_count'])
            c2.metric("Kháng sinh kháng", resistance_info['resistant_count'])

            # ==========================
            # TABLE 1: KHÁNG SINH NHẠY
            # ==========================
            if resistance_info['sensitive']:
                st.subheader("✅ Kháng Sinh Nhạy")
                df_sensitive = (
                    pd.DataFrame(resistance_info['sensitive'])[
                        ['name', 'code', 'sensitivity_probability', 'status']
                    ].rename(columns=COLUMN_MAP)
                )
                df_sensitive['Xác suất nhạy'] = df_sensitive['Xác suất nhạy'].apply(lambda x: f"{x:.3f}")
                # Dịch trạng thái sang tiếng Việt
                df_sensitive['Trạng thái'] = df_sensitive['Trạng thái'].replace({
                    'Sensitive': 'Nhạy',
                    'Resistant': 'Kháng'
                })
                st.markdown(render_centered_table(df_sensitive), unsafe_allow_html=True)

            # ==========================
            # TABLE 2: KHÁNG SINH KHÁNG
            # ==========================
            if resistance_info['resistant']:
                st.subheader("❌ Kháng Sinh Kháng")
                df_resistant = (
                    pd.DataFrame(resistance_info['resistant'])[
                        ['name', 'code', 'resistance_probability', 'status']
                    ].rename(columns=COLUMN_MAP)
                )
                df_resistant['Xác suất kháng'] = df_resistant['Xác suất kháng'].apply(lambda x: f"{x:.3f}")
                # Dịch trạng thái sang tiếng Việt
                df_resistant['Trạng thái'] = df_resistant['Trạng thái'].replace({
                    'Sensitive': 'Nhạy',
                    'Resistant': 'Kháng'
                })
                st.markdown(render_centered_table(df_resistant), unsafe_allow_html=True)

            # ==============================
            # TABLE 3: XÁC SUẤT TẤT CẢ KS
            # ==============================
            with st.expander("📈 Chi Tiết Xác Suất Tất Cả Kháng Sinh"):
                proba_series = pd.Series(result['probabilities'])
                st.bar_chart(proba_series, height=400)
                
                proba_table = pd.DataFrame({
                    'Mã kháng sinh': list(result['probabilities'].keys()),
                    'Xác suất nhạy/kháng': [f"{v:.3f}" for v in result['probabilities'].values()],
                    'Dự đoán': ['Nhạy' if result['predictions'][k] == 1 else 'Kháng'
                                for k in result['probabilities'].keys()]
                })
                st.markdown(render_centered_table(proba_table), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
            import traceback
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Hệ thống dự đoán kháng kháng sinh sử dụng Machine Learning</p>
</div>
""", unsafe_allow_html=True)
