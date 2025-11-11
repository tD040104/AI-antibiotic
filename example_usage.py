"""
Ví dụ sử dụng hệ thống Multi-Agent
"""

from main import MultiAgentOrchestrator
import pandas as pd


def example_train():
    """Ví dụ huấn luyện hệ thống"""
    print("=" * 80)
    print("VÍ DỤ: HUẤN LUYỆN HỆ THỐNG")
    print("=" * 80)
    
    orchestrator = MultiAgentOrchestrator(model_type='xgboost')
    
    # Huấn luyện
    results = orchestrator.train(
        'Bacteria_dataset_Multiresictance.csv',
        test_size=0.2,
        random_state=42
    )
    
    print("\nKết quả huấn luyện:")
    print(f"  - Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"  - Test Accuracy: {results['test_accuracy']:.4f}")
    
    return orchestrator


def example_predict(orchestrator):
    """Ví dụ dự đoán cho bệnh nhân mới"""
    print("\n" + "=" * 80)
    print("VÍ DỤ: DỰ ĐOÁN CHO BỆNH NHÂN MỚI")
    print("=" * 80)
    
    # Bệnh nhân mẫu 1: Nữ, 45 tuổi, E. coli, có tiểu đường và tiền sử nhập viện
    patient1 = {
        'age/gender': '45/F',
        'Souches': 'S999 Escherichia coli',
        'Diabetes': 'Yes',
        'Hypertension': 'No',
        'Hospital_before': 'Yes',
        'Infection_Freq': 2.0,
        'Collection_Date': '2024-01-15'
    }
    
    print("\nBệnh nhân 1:")
    print(f"  - Tuổi/Giới tính: {patient1['age/gender']}")
    print(f"  - Vi khuẩn: {patient1['Souches']}")
    print(f"  - Tiểu đường: {patient1['Diabetes']}")
    print(f"  - Tiền sử nhập viện: {patient1['Hospital_before']}")
    
    result1 = orchestrator.predict(patient1)
    
    print("\n" + "-" * 80)
    print("BÁO CÁO BỆNH NHÂN 1:")
    print("-" * 80)
    print(result1['report'])
    
    print("\n" + "-" * 80)
    print("KHUYẾN NGHỊ ĐIỀU TRỊ (Top 3):")
    print("-" * 80)
    for rec in result1['recommendations'][:3]:
        print(f"\n{rec['rank']}. {rec['antibiotic_name']}")
        print(f"   Mã: {rec['antibiotic_code']}")
        print(f"   Xác suất nhạy: {rec['sensitive_probability']:.1%}")
        print(f"   Độ tin cậy: {rec['confidence']}")
        print(f"   Nhóm: {rec['group']}")
    
    # Bệnh nhân mẫu 2: Nam, 68 tuổi, Klebsiella pneumoniae, tăng huyết áp
    patient2 = {
        'age/gender': '68/M',
        'Souches': 'S998 Klebsiella pneumoniae',
        'Diabetes': 'No',
        'Hypertension': 'Yes',
        'Hospital_before': 'No',
        'Infection_Freq': 1.0,
        'Collection_Date': '2024-02-20'
    }
    
    print("\n\n" + "=" * 80)
    print("Bệnh nhân 2:")
    print(f"  - Tuổi/Giới tính: {patient2['age/gender']}")
    print(f"  - Vi khuẩn: {patient2['Souches']}")
    print(f"  - Tăng huyết áp: {patient2['Hypertension']}")
    
    result2 = orchestrator.predict(patient2)
    
    print("\n" + "-" * 80)
    print("BÁO CÁO BỆNH NHÂN 2:")
    print("-" * 80)
    print(result2['report'])


def example_batch_predict(orchestrator):
    """Ví dụ dự đoán hàng loạt"""
    print("\n" + "=" * 80)
    print("VÍ DỤ: DỰ ĐOÁN HÀNG LOẠT")
    print("=" * 80)
    
    # Tạo danh sách bệnh nhân từ file CSV
    df = pd.read_csv('Bacteria_dataset_Multiresictance.csv')
    
    # Lấy 5 mẫu đầu tiên
    sample_patients = df.head(5)
    
    results = []
    for idx, row in sample_patients.iterrows():
        try:
            patient_data = {
                'age/gender': row.get('age/gender', ''),
                'Souches': row.get('Souches', ''),
                'Diabetes': 'Yes' if row.get('Diabetes') in ['Yes', 'True', True] else 'No',
                'Hypertension': 'Yes' if row.get('Hypertension') in ['Yes', 'True', True] else 'No',
                'Hospital_before': 'Yes' if row.get('Hospital_before') in ['Yes', 'True', True] else 'No',
                'Infection_Freq': row.get('Infection_Freq', 0.0),
                'Collection_Date': row.get('Collection_Date', '')
            }
            
            result = orchestrator.predict(patient_data)
            results.append({
                'patient_id': row.get('ID', idx),
                'recommendations': result['recommendations'][:2]  # Top 2
            })
        except Exception as e:
            print(f"Lỗi khi xử lý bệnh nhân {row.get('ID', idx)}: {e}")
    
    # In kết quả
    print(f"\nĐã xử lý {len(results)} bệnh nhân:")
    for res in results:
        print(f"\nBệnh nhân {res['patient_id']}:")
        for rec in res['recommendations']:
            print(f"  - {rec['antibiotic_name']} ({rec['sensitive_probability']:.1%})")


if __name__ == "__main__":
    # Huấn luyện hệ thống
    orchestrator = example_train()
    
    # Dự đoán cho bệnh nhân mới
    example_predict(orchestrator)
    
    # Dự đoán hàng loạt
    # example_batch_predict(orchestrator)




