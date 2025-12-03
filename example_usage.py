"""
Ví dụ sử dụng MASClinicalDecisionSystem với 5 agents
"""

from __future__ import annotations

import pandas as pd

from main import MASClinicalDecisionSystem


def example_train() -> MASClinicalDecisionSystem:
    print("=" * 80)
    print("HUẤN LUYỆN HỆ THỐNG MAS")
    print("=" * 80)

    system = MASClinicalDecisionSystem(model_type="xgboost")
    results = system.train(
        "Bacteria_dataset_Multiresictance.csv",
        test_size=0.2,
        random_state=42,
    )

    print("\nKết quả huấn luyện (Hamming accuracy):")
    print(f"  Train: {results['train_hamming_accuracy']:.4f}")
    print(f"  Test : {results['test_hamming_accuracy']:.4f}")
    return system


def example_predict(system: MASClinicalDecisionSystem):
    print("\n" + "=" * 80)
    print("DỰ ĐOÁN DỰA TRÊN 5 AGENTS")
    print("=" * 80)

    patient = {
        "age/gender": "60/F",
        "Souches": "S1200 Klebsiella pneumoniae",
        "Diabetes": "Yes",
        "Hypertension": "Yes",
        "Hospital_before": "No",
        "Infection_Freq": 3.0,
        "Collection_Date": "2024-03-01",
    }

    result = system.predict(patient)

    print("\n--- CHẤN ĐOÁN ---")
    for ab, label in result["predictions"].items():
        proba = result["probabilities"][ab]
        status = "Nhạy" if label == 1 else "Kháng"
        print(f"{ab:<15} -> {status} (P={proba:.2f})")

    print("\n--- PHÂN TÍCH CRITIC ---")
    critic = result["critic_report"]
    if critic["flags"]:
        for flag in critic["flags"]:
            print(f"{flag.antibiotic}: {flag.reason} (p={flag.probability:.2f})")
    else:
        print("Không có cảnh báo.")

    print("\n--- QUYẾT ĐỊNH LÂM SÀNG ---")
    for action in result["decision"]["primary_actions"]:
        print(" •", action)

    print("\n--- GỢI Ý KHÁNG SINH ---")
    for rec in result["decision"]["therapy_recommendations"][:3]:
        print(
            f"{rec['rank']}. {rec['antibiotic_name']} "
            f"(P={rec['sensitive_probability']:.2f}, {rec['confidence']})"
        )


def example_batch(system: MASClinicalDecisionSystem):
    print("\n" + "=" * 80)
    print("DỰ ĐOÁN HÀNG LOẠT")
    print("=" * 80)

    df = pd.read_csv("Bacteria_dataset_Multiresictance.csv").head(5)
    summaries = []

    for idx, row in df.iterrows():
        patient = {
            "age/gender": row.get("age/gender", ""),
            "Souches": row.get("Souches", ""),
            "Diabetes": "Yes" if row.get("Diabetes") in ["Yes", True, 1] else "No",
            "Hypertension": "Yes" if row.get("Hypertension") in ["Yes", True, 1] else "No",
            "Hospital_before": "Yes" if row.get("Hospital_before") in ["Yes", True, 1] else "No",
            "Infection_Freq": row.get("Infection_Freq", 0.0),
            "Collection_Date": row.get("Collection_Date", ""),
        }
        res = system.predict(patient)
        summaries.append(
            {
                "patient_id": row.get("ID", idx),
                "top_choice": res["decision"]["therapy_recommendations"][0]
                if res["decision"]["therapy_recommendations"]
                else None,
            }
        )

    for summary in summaries:
        choice = summary["top_choice"]
        if choice:
            print(
                f"Bệnh nhân {summary['patient_id']}: "
                f"{choice['antibiotic_name']} (P={choice['sensitive_probability']:.2f})"
            )


if __name__ == "__main__":
    system = example_train()
    example_predict(system)
    example_batch(system)