"""
Streamlit demo cho MASClinicalDecisionSystem (5 agents)
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MASClinicalDecisionSystem

# Page config
st.set_page_config(
    page_title="Dự Đoán Kháng Kháng Sinh",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 Hệ Thống Đa Tác Nhân Kháng Kháng Sinh")
st.markdown("---")

st.markdown("""
<style>
    div[data-testid="stDataFrame"] table th,
    div[data-testid="stDataFrame"] table td {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

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

COLUMN_MAP = {
    'name': 'Tên kháng sinh',
    'code': 'Mã',
    'sensitivity_probability': 'Xác suất nhạy',
    'resistance_probability': 'Xác suất kháng',
    'status': 'Trạng thái'
}

ANTIBIOTIC_NAME_MAP = {
    'AMX/AMP': 'Amoxicillin/Ampicillin',
    'AMC': 'Amoxicillin-Clavulanic Acid',
    'CZ': 'Cefazolin',
    'FOX': 'Cefoxitin',
    'CTX/CRO': 'Ceftriaxone/Cefotaxime',
    'IPM': 'Imipenem',
    'GEN': 'Gentamicin',
    'AN': 'Amikacin',
    'Acide nalidixique': 'Nalidixic Acid',
    'ofx': 'Ofloxacin',
    'CIP': 'Ciprofloxacin',
    'C': 'Chloramphenicol',
    'Co-trimoxazole': 'Trimethoprim-Sulfamethoxazole',
    'Furanes': 'Nitrofurantoin',
    'colistine': 'Colistin'
}


def render_centered_table(df: pd.DataFrame) -> str:
    return (
        "<div style='overflow-x:auto;'>"
        + df.to_html(index=False, justify="center")
        + "</div>"
    )


if 'mas_system' not in st.session_state:
    st.session_state.mas_system = None
    st.session_state.model_loaded = False

with st.sidebar:
    st.header("⚙️ Cài Đặt")
    model_path = st.text_input("Đường dẫn mô hình", value="models/mas_model.pkl")
    state_path = st.text_input("Đường dẫn trạng thái", value="models/mas_state.joblib")

    if st.button("📥 Tải Mô Hình", type="primary"):
        try:
            system = MASClinicalDecisionSystem()
            system.load(model_path=model_path, state_path=state_path)
            st.session_state.mas_system = system
            st.session_state.model_loaded = True
            st.success("✅ Đã tải mô hình MAS thành công!")
        except Exception as exc:
            st.error(f"❌ Lỗi khi tải mô hình: {exc}")
            st.session_state.mas_system = None
            st.session_state.model_loaded = False

    if st.session_state.model_loaded:
        st.success("✅ Mô hình đã sẵn sàng")
    else:
        st.warning("⚠️ Vui lòng tải mô hình trước khi dự đoán")

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
        # Tần suất nhiễm trùng: mỗi lần bấm "+" tăng 1 đơn vị, không vượt quá 3
        infection_freq = st.number_input(
            "Tần suất nhiễm trùng",
            min_value=0,
            max_value=3,
            value=1,
            step=1,
        )
        collection_date = st.date_input("Ngày thu thập mẫu", value=pd.Timestamp.now().date())
        hypertension = st.selectbox("Tăng huyết áp", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Dự Đoán", type="primary", use_container_width=True)

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
                result = st.session_state.mas_system.predict(patient_data)

            st.success("✅ Hoàn tất dự đoán!")
            st.markdown("---")

            predictions = result['predictions']
            probabilities = result['probabilities']
            sensitive_entries = []
            resistant_entries = []
            for code, label in predictions.items():
                name = ANTIBIOTIC_NAME_MAP.get(code, code)
                proba = probabilities.get(code, 0.0)
                if label == 1:
                    sensitive_entries.append({
                        'name': name,
                        'code': code,
                        'sensitivity_probability': proba,
                        'status': 'Sensitive'
                    })
                else:
                    resistant_entries.append({
                        'name': name,
                        'code': code,
                        'resistance_probability': 1 - proba,
                        'status': 'Resistant'
                    })

            st.header("📊 Kết Quả Dự Đoán")
            c1, c2 = st.columns(2)
            c1.metric("Kháng sinh nhạy", len(sensitive_entries))
            c2.metric("Kháng sinh kháng", len(resistant_entries))

            if sensitive_entries:
                st.subheader("✅ Kháng Sinh Nhạy")
                df_sensitive = pd.DataFrame(sensitive_entries)[['name', 'code', 'sensitivity_probability', 'status']]
                df_sensitive = df_sensitive.rename(columns=COLUMN_MAP)
                df_sensitive['Xác suất nhạy'] = df_sensitive['Xác suất nhạy'].apply(lambda x: f"{x:.3f}")
                df_sensitive['Trạng thái'] = df_sensitive['Trạng thái'].replace({'Sensitive': 'Nhạy'})
                st.markdown(render_centered_table(df_sensitive), unsafe_allow_html=True)
            else:
                st.info("Không có kháng sinh nào được dự đoán nhạy.")

            if resistant_entries:
                st.subheader("❌ Kháng Sinh Kháng")
                df_resistant = pd.DataFrame(resistant_entries)[['name', 'code', 'resistance_probability', 'status']]
                df_resistant = df_resistant.rename(columns=COLUMN_MAP)
                df_resistant['Xác suất kháng'] = df_resistant['Xác suất kháng'].apply(lambda x: f"{x:.3f}")
                df_resistant['Trạng thái'] = df_resistant['Trạng thái'].replace({'Resistant': 'Kháng'})
                st.markdown(render_centered_table(df_resistant), unsafe_allow_html=True)
            else:
                st.success("Tuyệt vời! Không có kháng sinh nào bị dự đoán kháng.")

            with st.expander("📈 Xác Suất Chi Tiết"):
                proba_series = pd.Series(probabilities)
                st.bar_chart(proba_series, height=400)
                proba_table = pd.DataFrame({
                    'Mã kháng sinh': list(probabilities.keys()),
                    'Xác suất nhạy': [f"{v:.3f}" for v in probabilities.values()],
                    'Dự đoán': ['Nhạy' if predictions[k] == 1 else 'Kháng' for k in probabilities.keys()]
                })
                st.markdown(render_centered_table(proba_table), unsafe_allow_html=True)

            st.markdown("---")
            st.header("🕵️‍♂️ Critic Agent")
            critic_report = result.get('critic_report', {})
            flags = critic_report.get('flags', [])
            missing_fields = critic_report.get('missing_fields', [])

            if flags:
                st.warning("Một số kháng sinh có xác suất không chắc chắn:")
                for flag in flags:
                    st.write(f"- {flag.antibiotic}: p={flag.probability:.2f} ({flag.reason})")
            else:
                st.success("Critic Agent: Không có cảnh báo về độ chắc chắn.")

            if missing_fields:
                st.info("Thiếu dữ liệu ở các trường: " + ", ".join(missing_fields))

            st.markdown("---")
            st.header("🧠 Decision Agent")
            decision = result.get('decision', {})
            actions = decision.get('primary_actions', [])
            recommendations = decision.get('therapy_recommendations', [])

            if actions:
                st.subheader("Hành động ưu tiên")
                for action in actions:
                    st.write(f"- {action}")

            if recommendations:
                st.subheader("Khuyến nghị kháng sinh")
                for rec in recommendations[:5]:
                    st.write(
                        f"{rec['rank']}. {rec['antibiotic_name']} "
                        f"(Mã: {rec['antibiotic_code']}, "
                        f"P={rec['sensitive_probability']:.2f}, "
                        f"Độ tin cậy: {rec['confidence']})"
                    )
            else:
                st.warning("Chưa có khuyến nghị điều trị rõ ràng.")

            explanation = result.get('explanation', {})
            if explanation.get('report'):
                st.markdown("---")
                st.header("📝 Tóm Tắt Giải Thích")
                st.write(explanation['report'])

        except Exception as exc:
            st.error(f"❌ Lỗi khi dự đoán: {exc}")
            import traceback
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Hệ thống đa tác nhân MAS cho phân tích kháng sinh"
    "</div>",
    unsafe_allow_html=True,
)
