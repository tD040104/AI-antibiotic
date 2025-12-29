"""
Hệ thống đánh giá và so sánh 2 mô hình decision:
1. Vector (ML) - sử dụng decision tree và fuzzy logic với vector inputs
2. Gemini (LLM) - sử dụng Gemini API để đưa ra quyết định
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import json
import time
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, try manual loading
    try:
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8-sig') as f:  # utf-8-sig removes BOM
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    except Exception:
        pass
except Exception:
    # If load_dotenv fails, try manual loading
    try:
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8-sig') as f:  # utf-8-sig removes BOM
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    except Exception:
        pass

from main import MASClinicalDecisionSystem
from agents.decision_agent import DecisionAgent, LLMDecisionMaker


class DecisionModelEvaluator:
    """Đánh giá và so sánh 2 mô hình decision: Vector (ML) và Gemini (LLM)"""
    
    def __init__(
        self,
        system: MASClinicalDecisionSystem,
        gemini_api_key: Optional[str] = None,
        enable_vector: bool = True,
        enable_llm: bool = True
    ):
        """
        Args:
            system: MASClinicalDecisionSystem đã được train
            gemini_api_key: API key cho Gemini (nếu dùng LLM)
            enable_vector: Enable Vector (ML) model
            enable_llm: Enable Gemini (LLM) model
        """
        self.system = system
        self.enable_vector = enable_vector
        self.enable_llm = enable_llm
        
        # Tạo decision agents dựa trên flags
        self.vector_agent = None
        self.gemini_agent = None
        
        if enable_vector:
            # Agent 1: Chỉ dùng Vector (ML) - không dùng LLM
            self.vector_agent = DecisionAgent(
                use_fuzzy=True,
                use_decision_tree=True,
                use_llm=False,
                decision_mode="vector_only"  # Chỉ dùng vector
            )
            print(f"  ✓ Vector (ML) model đã được kích hoạt")
        
        if enable_llm:
            # Agent 2: Chỉ dùng Gemini (LLM)
            self.gemini_agent = DecisionAgent(
                use_fuzzy=True,
                use_decision_tree=True,
                use_llm=True,
                llm_api_key=gemini_api_key,
                decision_mode="llm_only"  # Chỉ dùng LLM
            )
        
            # Kiểm tra xem LLM có available không
            if self.gemini_agent.llm_decision_maker:
                if self.gemini_agent.llm_decision_maker.available:
                    print(f"  ✓ Gemini LLM đã được kích hoạt")
                    # Test LLM với một sample đơn giản
                    try:
                        test_result = self.gemini_agent.llm_decision_maker.generate_decision(
                            patient_data={"age": 50, "gender": "M"},
                            probabilities={"AMX/AMP": 0.8, "CIP": 0.6},
                            critic_report={"decision": {"uncertainty_score": 0.2}},
                            explanation={"decision": {"decision_score": 0.7}}
                        )
                        if test_result.get("decision_type") not in ["llm_unavailable", "llm_error"]:
                            print(f"  ✓ Test LLM thành công: {test_result.get('decision_type')}")
                        else:
                            print(f"  ⚠️  Test LLM thất bại: {test_result.get('decision_type')}")
                            if test_result.get("error"):
                                print(f"     Error: {test_result.get('error')}")
                    except Exception as e:
                        print(f"  ⚠️  Test LLM exception: {str(e)}")
                else:
                    print(f"  ⚠️  Gemini LLM không available (kiểm tra API key)")
            else:
                print(f"  ⚠️  LLMDecisionMaker chưa được khởi tạo")
        
        if not enable_vector and not enable_llm:
            raise ValueError("Phải enable ít nhất 1 mô hình (enable_vector hoặc enable_llm)")
    
    def _normalize_decision_type(self, decision_type: str) -> str:
        """Chuẩn hóa decision_type về 3 loại: treat, review, test"""
        decision_lower = decision_type.lower()
        
        if "treat" in decision_lower or "high_confidence" in decision_lower:
            return "treat"
        elif "test" in decision_lower or "additional" in decision_lower or "requires" in decision_lower:
            return "test"
        else:
            return "review"
    
    def _get_ground_truth_decision(
        self,
        probabilities: pd.DataFrame,
        patient_features: Optional[pd.Series],
        critic_report: Dict
    ) -> str:
        """
        Tạo ground truth decision dựa trên heuristic cải tiến:
        - treat: nếu có antibiotic với prob rất cao (>= 0.8) hoặc prob cao (>= 0.7) với uncertainty thấp
        - test: nếu không có antibiotic nào tốt hoặc uncertainty rất cao
        - review: các trường hợp trung gian
        """
        if probabilities.empty:
            return "test"
        
        proba_series = probabilities.iloc[0]
        max_prob = proba_series.max()
        mean_prob = proba_series.mean()
        
        # Đếm số antibiotic có prob >= 0.5, >= 0.7, và >= 0.8
        very_high_prob_count = (proba_series >= 0.8).sum()
        high_prob_count = (proba_series >= 0.7).sum()
        medium_prob_count = (proba_series >= 0.5).sum()
        
        uncertainty = critic_report.get("decision", {}).get("uncertainty_score", 0.5)
        risk_factors = patient_features.get("Total_risk_factors", 0) if patient_features is not None else 0
        
        # Heuristic rules cải tiến - ưu tiên max_prob cao
        # TREAT: 
        # 1. Có ít nhất 1 antibiotic với prob rất cao (>= 0.8) và uncertainty không quá cao
        if max_prob >= 0.8 and uncertainty < 0.6 and risk_factors <= 3:
            return "treat"
        # 2. Có ít nhất 1 antibiotic với prob cao (>= 0.75) và uncertainty thấp
        elif max_prob >= 0.75 and uncertainty < 0.35 and risk_factors <= 2:
            return "treat"
        # 3. Có nhiều antibiotic với prob tốt (>= 0.7) và uncertainty thấp
        elif high_prob_count >= 2 and uncertainty < 0.4 and risk_factors <= 2:
            return "treat"
        
        # TEST: 
        # 1. Không có antibiotic nào tốt
        elif max_prob < 0.4 or medium_prob_count == 0:
            return "test"
        # 2. Uncertainty rất cao
        elif uncertainty > 0.75:
            return "test"
        # 3. Risk rất cao
        elif risk_factors >= 5:
            return "test"
        
        # REVIEW: Các trường hợp trung gian
        # 1. Có antibiotic với prob tốt nhưng uncertainty trung bình-cao
        elif max_prob >= 0.7 and uncertainty >= 0.4 and uncertainty < 0.7:
            return "review"
        # 2. Có antibiotic với prob trung bình (0.5-0.7)
        elif max_prob >= 0.5 and max_prob < 0.7:
            return "review"
        # 3. Có ít antibiotic với prob tốt nhưng uncertainty trung bình
        elif medium_prob_count >= 1 and medium_prob_count < 3 and uncertainty < 0.6:
            return "review"
        # Mặc định: review cho các trường hợp không rõ ràng
        else:
            return "review"
    
    def evaluate_on_dataset(
        self,
        csv_path: str,
        n_samples: Optional[int] = None,
        use_ground_truth: bool = True,
        llm_delay: float = 0.0
    ) -> Dict:
        """
        Đánh giá 2 mô hình trên dataset.
        
        Args:
            csv_path: Đường dẫn đến file CSV
            n_samples: Số lượng mẫu để đánh giá (None = tất cả)
            use_ground_truth: Nếu True, sử dụng ground truth heuristic. Nếu False, chỉ so sánh 2 mô hình với nhau.
            llm_delay: Thời gian delay (giây) giữa mỗi lần gọi LLM để tránh quota. Mặc định 0.0 (không delay).
        """
        print("=" * 80)
        print("BẮT ĐẦU ĐÁNH GIÁ 2 MÔ HÌNH DECISION")
        print("=" * 80)
        
        # Đọc dữ liệu
        df = pd.read_csv(csv_path)
        if n_samples:
            df = df.head(n_samples)
        
        print(f"  ✓ Đã tải {len(df)} mẫu từ dataset")
        
        # Thông báo về delay nếu có
        if llm_delay > 0 and self.enable_llm:
            total_delay_time = llm_delay * (len(df) - 1) if len(df) > 1 else 0
            print(f"  ⏳ LLM delay: {llm_delay}s giữa mỗi mẫu (tổng thời gian delay: ~{total_delay_time:.0f}s)")
        
        # Kết quả - chỉ khởi tạo cho các mô hình được enable
        vector_predictions = []
        gemini_predictions = []
        ground_truths = []
        vector_scores = []
        gemini_scores = []
        vector_methods = []
        gemini_methods = []
        
        # Kiểm tra có ít nhất 1 mô hình được enable
        if not self.enable_vector and not self.enable_llm:
            raise ValueError("Phải enable ít nhất 1 mô hình để đánh giá")
        
        print("\n  Đang chạy đánh giá trên từng mẫu...")
        
        for idx, row in df.iterrows():
            if (idx + 1) % 10 == 0:
                print(f"    Đã xử lý {idx + 1}/{len(df)} mẫu...")
            
            try:
                # Chuẩn bị dữ liệu bệnh nhân
                patient = {
                    "age/gender": row.get("age/gender", ""),
                    "Souches": row.get("Souches", ""),
                    "Diabetes": "Yes" if row.get("Diabetes") in ["Yes", True, 1] else "No",
                    "Hypertension": "Yes" if row.get("Hypertension") in ["Yes", True, 1] else "No",
                    "Hospital_before": "Yes" if row.get("Hospital_before") in ["Yes", True, 1] else "No",
                    "Infection_Freq": row.get("Infection_Freq", 0.0),
                    "Collection_Date": row.get("Collection_Date", ""),
                }
                
                # Chạy pipeline để lấy các inputs cần thiết
                result = self.system.predict(patient)
                
                # Chuyển đổi dict thành DataFrame/Series đúng cách
                # probabilities và predictions từ result là dict, cần chuyển thành DataFrame
                probabilities_dict = result["probabilities"]
                predictions_dict = result["predictions"]
                
                # Tạo DataFrame với index [0] để có 1 row
                # Đảm bảo DataFrame có đúng cấu trúc
                probabilities = pd.DataFrame([probabilities_dict], index=[0])
                predictions = pd.DataFrame([predictions_dict], index=[0])
                patient_series = pd.Series(result["features"])
                
                # Kiểm tra DataFrame không rỗng và có dữ liệu hợp lệ
                if probabilities.empty or len(probabilities) == 0:
                    print(f"    ⚠️  Probabilities rỗng cho mẫu {idx}")
                    continue
                
                # Đảm bảo predictions và probabilities có cùng số cột
                if len(predictions.columns) == 0 or len(probabilities.columns) == 0:
                    print(f"    ⚠️  Dữ liệu không hợp lệ cho mẫu {idx}")
                    continue
                
                critic_report = result["critic_report"]
                explanation = result["explanation"]
                
                # Lấy vectors từ Agent 3 và Agent 4
                # explain_vector và review_vector cần DataFrame, không phải Series
                try:
                    explain_vector = self.system.pipeline.explain_agent.explain_vector(
                        patient_series,
                        predictions,  # DataFrame, không phải Series
                        probabilities  # DataFrame, không phải Series
                    )
                    critic_vector = self.system.pipeline.critic_agent.review_vector(
                        probabilities,
                        patient_series
                    )
                except Exception as vec_error:
                    print(f"    ⚠️  Lỗi khi tạo vectors cho mẫu {idx}: {str(vec_error)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # 1. Chạy Vector (ML) model (nếu được enable)
                if self.enable_vector and self.vector_agent:
                    vector_decision = self.vector_agent.decide(
                        probabilities,
                        critic_report,
                        patient_series,
                        explanation=explanation,
                        explain_vector=explain_vector,
                        critic_vector=critic_vector
                    )
                    
                    vector_decision_type = self._normalize_decision_type(
                        vector_decision["decision"].get("decision_type", "review")
                    )
                    vector_predictions.append(vector_decision_type)
                    vector_scores.append(vector_decision["decision"].get("decision_score", 0.5))
                    vector_methods.append(vector_decision["decision"].get("method", "unknown"))
                
                # 2. Chạy Gemini (LLM) model (nếu được enable)
                if self.enable_llm and self.gemini_agent:
                    # Không truyền vectors cho LLM-only mode để force sử dụng LLM
                    gemini_decision = self.gemini_agent.decide(
                        probabilities,
                        critic_report,
                        patient_series,
                        explanation=explanation,
                        explain_vector=None,  # Không truyền vectors để force LLM
                        critic_vector=None
                    )
                    
                    raw_decision_type = gemini_decision["decision"].get("decision_type", "review")
                    gemini_decision_type = self._normalize_decision_type(raw_decision_type)
                    gemini_predictions.append(gemini_decision_type)
                    gemini_scores.append(gemini_decision["decision"].get("decision_score", 0.5))
                    gemini_method = gemini_decision["decision"].get("method", "unknown")
                    gemini_methods.append(gemini_method)
                    
                    # Debug: Log raw decision type và normalized cho vài mẫu đầu
                    if idx < 5:
                        print(f"    DEBUG Sample {idx+1}: raw_decision='{raw_decision_type}' -> normalized='{gemini_decision_type}', method='{gemini_method}'")
                    
                    # Debug: Log nếu không dùng LLM (chỉ log 1 lần đầu tiên)
                    if idx == 0 and "llm" not in gemini_method.lower() and "gemini" not in gemini_method.lower():
                        print(f"\n    ⚠️  DEBUG: Gemini model đang dùng method '{gemini_method}' thay vì LLM")
                        if gemini_decision["decision"].get("error"):
                            print(f"    ⚠️  LLM Error: {gemini_decision['decision'].get('error')}")
                    
                    # Delay giữa các lần gọi LLM để tránh quota (chỉ delay nếu dùng LLM và không phải mẫu cuối)
                    if (llm_delay > 0 and 
                        self.enable_llm and 
                        self.gemini_agent and 
                        self.gemini_agent.llm_decision_maker and
                        self.gemini_agent.llm_decision_maker.available and
                        "llm" in gemini_method.lower() and 
                        idx < len(df) - 1):
                        print(f"    ⏳ Đang delay {llm_delay}s trước mẫu tiếp theo...")
                        time.sleep(llm_delay)
                
                # 3. Ground truth (nếu cần) - tính cho mỗi mẫu đã xử lý thành công
                # Ground truth phải được tính cho mỗi mẫu, không phụ thuộc vào số lượng predictions
                if use_ground_truth:
                    # Tính ground truth dựa trên số mẫu đã xử lý (tối đa là số predictions đã có)
                    # Đảm bảo mỗi mẫu chỉ có 1 ground truth
                    current_sample_idx = max(len(vector_predictions), len(gemini_predictions), len(ground_truths))
                    if len(ground_truths) < current_sample_idx:
                        gt = self._get_ground_truth_decision(
                            probabilities,
                            patient_series,
                            critic_report
                        )
                        ground_truths.append(gt)
                        
                        # Debug: Log thông tin cho vài mẫu đầu tiên
                        if len(ground_truths) <= 5:
                            proba_series = probabilities.iloc[0]
                            max_prob = proba_series.max()
                            uncertainty = critic_report.get("decision", {}).get("uncertainty_score", 0.5)
                            risk_factors = patient_series.get("Total_risk_factors", 0) if patient_series is not None else 0
                            print(f"    DEBUG Sample {len(ground_truths)}: max_prob={max_prob:.3f}, uncertainty={uncertainty:.3f}, risk={risk_factors}, GT={gt}")
                
            except Exception as e:
                print(f"    ⚠️  Lỗi khi xử lý mẫu {idx}: {str(e)}")
                continue
        
        # Tính số mẫu đã xử lý và đảm bảo độ dài khớp nhau
        n_processed = max(len(vector_predictions), len(gemini_predictions))
        
        # Đảm bảo ground_truths có cùng độ dài với số mẫu đã xử lý
        if use_ground_truth:
            while len(ground_truths) < n_processed:
                # Nếu thiếu ground truth, thêm giá trị mặc định (không nên xảy ra)
                print(f"  ⚠️  CẢNH BÁO: Thiếu ground truth cho {n_processed - len(ground_truths)} mẫu")
                break
            # Cắt bớt nếu thừa
            if len(ground_truths) > n_processed:
                ground_truths = ground_truths[:n_processed]
                print(f"  ⚠️  CẢNH BÁO: Đã cắt bớt ground_truths từ {len(ground_truths)} xuống {n_processed}")
        
        # Đảm bảo vector_predictions và gemini_predictions có cùng độ dài
        if self.enable_vector and self.enable_llm:
            min_len = min(len(vector_predictions), len(gemini_predictions))
            if len(vector_predictions) != len(gemini_predictions):
                print(f"  ⚠️  CẢNH BÁO: Vector có {len(vector_predictions)} predictions, Gemini có {len(gemini_predictions)} predictions")
                # Cắt về độ dài nhỏ nhất để đảm bảo tính metrics chính xác
                vector_predictions = vector_predictions[:min_len]
                gemini_predictions = gemini_predictions[:min_len]
                if use_ground_truth:
                    ground_truths = ground_truths[:min_len]
                n_processed = min_len
        
        print(f"\n  ✓ Đã hoàn thành đánh giá {n_processed} mẫu")
        if use_ground_truth:
            print(f"  ✓ Số lượng ground_truths: {len(ground_truths)}")
        if self.enable_vector:
            print(f"  ✓ Số lượng vector_predictions: {len(vector_predictions)}")
        if self.enable_llm:
            print(f"  ✓ Số lượng gemini_predictions: {len(gemini_predictions)}")
        
        # Debug: Kiểm tra phân bố predictions
        if self.enable_llm and len(gemini_predictions) > 0:
            from collections import Counter
            gemini_dist = Counter(gemini_predictions)
            print(f"\n  📊 Phân bố Gemini predictions: {dict(gemini_dist)}")
            if len(gemini_dist) == 1:
                print(f"  ⚠️  CẢNH BÁO: Tất cả Gemini predictions đều giống nhau: {list(gemini_dist.keys())[0]}")
        
        if use_ground_truth and len(ground_truths) > 0:
            from collections import Counter
            gt_dist = Counter(ground_truths)
            print(f"  📊 Phân bố Ground Truth: {dict(gt_dist)}")
            if len(gt_dist) == 1:
                print(f"  ⚠️  CẢNH BÁO: Tất cả Ground Truth đều giống nhau: {list(gt_dist.keys())[0]}")
            
            # Phân tích chi tiết hơn
            if self.enable_vector and len(vector_predictions) > 0:
                vector_dist = Counter(vector_predictions)
                print(f"  📊 Phân bố Vector predictions: {dict(vector_dist)}")
                # So sánh với ground truth
                matches = sum(1 for gt, pred in zip(ground_truths, vector_predictions) if gt == pred)
                print(f"  📊 Vector matches GT: {matches}/{len(ground_truths)} ({matches/len(ground_truths)*100:.1f}%)")
            
            if self.enable_llm and len(gemini_predictions) > 0:
                gemini_dist = Counter(gemini_predictions)
                print(f"  📊 Phân bố Gemini predictions: {dict(gemini_dist)}")
                # So sánh với ground truth
                matches = sum(1 for gt, pred in zip(ground_truths, gemini_predictions) if gt == pred)
                print(f"  📊 Gemini matches GT: {matches}/{len(ground_truths)} ({matches/len(ground_truths)*100:.1f}%)")
        
        # Tính metrics
        results = {
            "n_samples": n_processed,
            "enable_vector": self.enable_vector,
            "enable_llm": self.enable_llm,
            "vector_predictions": vector_predictions if self.enable_vector else [],
            "gemini_predictions": gemini_predictions if self.enable_llm else [],
            "vector_scores": vector_scores if self.enable_vector else [],
            "gemini_scores": gemini_scores if self.enable_llm else [],
            "vector_methods": vector_methods if self.enable_vector else [],
            "gemini_methods": gemini_methods if self.enable_llm else [],
        }
        
        if use_ground_truth:
            results["ground_truths"] = ground_truths
            results["metrics"] = self._calculate_metrics_with_ground_truth(
                ground_truths,
                vector_predictions if self.enable_vector else [],
                gemini_predictions if self.enable_llm else []
            )
        else:
            results["metrics"] = self._calculate_metrics_comparison(
                vector_predictions if self.enable_vector else [],
                gemini_predictions if self.enable_llm else [],
                vector_scores if self.enable_vector else [],
                gemini_scores if self.enable_llm else []
            )
        
        return results
    
    def _calculate_metrics_with_ground_truth(
        self,
        ground_truths: List[str],
        vector_predictions: List[str],
        gemini_predictions: List[str]
    ) -> Dict:
        """Tính metrics khi có ground truth"""
        labels = ["treat", "review", "test"]
        metrics = {}
        
        # Validation: Đảm bảo độ dài khớp nhau
        if self.enable_vector and len(vector_predictions) > 0:
            if len(ground_truths) != len(vector_predictions):
                print(f"  ⚠️  LỖI: ground_truths ({len(ground_truths)}) và vector_predictions ({len(vector_predictions)}) có độ dài khác nhau!")
                min_len = min(len(ground_truths), len(vector_predictions))
                ground_truths = ground_truths[:min_len]
                vector_predictions = vector_predictions[:min_len]
                print(f"  ⚠️  Đã cắt về độ dài {min_len} để tính metrics")
        
        if self.enable_llm and len(gemini_predictions) > 0:
            if len(ground_truths) != len(gemini_predictions):
                print(f"  ⚠️  LỖI: ground_truths ({len(ground_truths)}) và gemini_predictions ({len(gemini_predictions)}) có độ dài khác nhau!")
                min_len = min(len(ground_truths), len(gemini_predictions))
                ground_truths = ground_truths[:min_len]
                gemini_predictions = gemini_predictions[:min_len]
                print(f"  ⚠️  Đã cắt về độ dài {min_len} để tính metrics")
        
        # Metrics cho Vector model (nếu được enable)
        if self.enable_vector and len(vector_predictions) > 0:
            # Debug: In ra một vài ví dụ để kiểm tra
            if len(vector_predictions) <= 10:
                print(f"\n  🔍 DEBUG Vector Model:")
                print(f"    Ground Truth: {ground_truths}")
                print(f"    Predictions:  {vector_predictions}")
                matches = sum(1 for gt, pred in zip(ground_truths, vector_predictions) if gt == pred)
                print(f"    Matches: {matches}/{len(ground_truths)}")
            
            vector_accuracy = accuracy_score(ground_truths, vector_predictions)
            vector_precision = precision_score(ground_truths, vector_predictions, labels=labels, average="weighted", zero_division=0)
            vector_recall = recall_score(ground_truths, vector_predictions, labels=labels, average="weighted", zero_division=0)
            vector_f1 = f1_score(ground_truths, vector_predictions, labels=labels, average="weighted", zero_division=0)
            vector_cm = confusion_matrix(ground_truths, vector_predictions, labels=labels)
            vector_report = classification_report(ground_truths, vector_predictions, labels=labels, output_dict=True, zero_division=0)
            
            metrics["vector_model"] = {
                "accuracy": float(vector_accuracy),
                "precision": float(vector_precision),
                "recall": float(vector_recall),
                "f1_score": float(vector_f1),
                "confusion_matrix": vector_cm.tolist(),
                "classification_report": vector_report
            }
        
        # Metrics cho Gemini model (nếu được enable)
        if self.enable_llm and len(gemini_predictions) > 0:
            # Debug: In ra một vài ví dụ để kiểm tra
            if len(gemini_predictions) <= 10:
                print(f"\n  🔍 DEBUG Gemini Model:")
                print(f"    Ground Truth: {ground_truths}")
                print(f"    Predictions:  {gemini_predictions}")
                matches = sum(1 for gt, pred in zip(ground_truths, gemini_predictions) if gt == pred)
                print(f"    Matches: {matches}/{len(ground_truths)}")
                # In chi tiết từng cặp để debug
                print(f"    Chi tiết so sánh:")
                for i, (gt, pred) in enumerate(zip(ground_truths, gemini_predictions)):
                    match_symbol = "✓" if gt == pred else "✗"
                    print(f"      Sample {i+1}: GT={gt:6} vs Pred={pred:6} {match_symbol}")
            
            # Kiểm tra phân bố classes
            from collections import Counter
            gt_dist = Counter(ground_truths)
            pred_dist = Counter(gemini_predictions)
            
            # Cảnh báo nếu chỉ có một class duy nhất
            if len(gt_dist) == 1:
                print(f"  ⚠️  CẢNH BÁO: Ground truth chỉ có 1 class duy nhất: {list(gt_dist.keys())[0]}")
            if len(pred_dist) == 1:
                print(f"  ⚠️  CẢNH BÁO: Gemini predictions chỉ có 1 class duy nhất: {list(pred_dist.keys())[0]}")
            
            gemini_accuracy = accuracy_score(ground_truths, gemini_predictions)
            gemini_precision = precision_score(ground_truths, gemini_predictions, labels=labels, average="weighted", zero_division=0)
            gemini_recall = recall_score(ground_truths, gemini_predictions, labels=labels, average="weighted", zero_division=0)
            gemini_f1 = f1_score(ground_truths, gemini_predictions, labels=labels, average="weighted", zero_division=0)
            gemini_cm = confusion_matrix(ground_truths, gemini_predictions, labels=labels)
            gemini_report = classification_report(ground_truths, gemini_predictions, labels=labels, output_dict=True, zero_division=0)
            
            # Cảnh báo nếu tất cả metrics đều bằng 1.0
            if gemini_accuracy == 1.0 and gemini_precision == 1.0 and gemini_recall == 1.0 and gemini_f1 == 1.0:
                print(f"  ℹ️  THÔNG TIN: Tất cả metrics đều bằng 1.0 - Đây là kết quả ĐÚNG khi:")
                print(f"      ✓ Tất cả predictions đều khớp với ground truth (100% accuracy)")
                print(f"      ✓ Không có false positives (100% precision)")
                print(f"      ✓ Không có false negatives (100% recall)")
                print(f"      - Phân bố GT: {dict(gt_dist)}")
                print(f"      - Phân bố Predictions: {dict(pred_dist)}")
                if len(gemini_predictions) < 20:
                    print(f"  ⚠️  LƯU Ý: Với chỉ {len(gemini_predictions)} mẫu, kết quả có thể không đại diện.")
                    print(f"      Khuyến nghị: Tăng số lượng mẫu lên ít nhất 20-50 để đánh giá đáng tin cậy hơn.")
            
            metrics["gemini_model"] = {
                "accuracy": float(gemini_accuracy),
                "precision": float(gemini_precision),
                "recall": float(gemini_recall),
                "f1_score": float(gemini_f1),
                "confusion_matrix": gemini_cm.tolist(),
                "classification_report": gemini_report
            }
        
        # So sánh (chỉ khi cả 2 đều được enable)
        if self.enable_vector and self.enable_llm and len(vector_predictions) > 0 and len(gemini_predictions) > 0:
            vector_accuracy = metrics["vector_model"]["accuracy"]
            gemini_accuracy = metrics["gemini_model"]["accuracy"]
            vector_f1 = metrics["vector_model"]["f1_score"]
            gemini_f1 = metrics["gemini_model"]["f1_score"]
            vector_precision = metrics["vector_model"]["precision"]
            gemini_precision = metrics["gemini_model"]["precision"]
            vector_recall = metrics["vector_model"]["recall"]
            gemini_recall = metrics["gemini_model"]["recall"]
            
            metrics["comparison"] = {
                "accuracy_diff": float(gemini_accuracy - vector_accuracy),
                "precision_diff": float(gemini_precision - vector_precision),
                "recall_diff": float(gemini_recall - vector_recall),
                "f1_diff": float(gemini_f1 - vector_f1),
                "winner_accuracy": "gemini" if gemini_accuracy > vector_accuracy else "vector",
                "winner_f1": "gemini" if gemini_f1 > vector_f1 else "vector"
            }
        
        return metrics
    
    def _calculate_metrics_comparison(
        self,
        vector_predictions: List[str],
        gemini_predictions: List[str],
        vector_scores: List[float],
        gemini_scores: List[float]
    ) -> Dict:
        """Tính metrics khi so sánh trực tiếp 2 mô hình (không có ground truth)"""
        # Agreement rate
        agreement = sum(1 for v, g in zip(vector_predictions, gemini_predictions) if v == g)
        agreement_rate = agreement / len(vector_predictions) if vector_predictions else 0
        
        # Score statistics
        vector_mean_score = np.mean(vector_scores) if vector_scores else 0
        gemini_mean_score = np.mean(gemini_scores) if gemini_scores else 0
        vector_std_score = np.std(vector_scores) if vector_scores else 0
        gemini_std_score = np.std(gemini_scores) if gemini_scores else 0
        
        # Distribution of decisions
        from collections import Counter
        vector_dist = Counter(vector_predictions)
        gemini_dist = Counter(gemini_predictions)
        
        return {
            "agreement_rate": float(agreement_rate),
            "vector_model": {
                "mean_score": float(vector_mean_score),
                "std_score": float(vector_std_score),
                "decision_distribution": dict(vector_dist)
            },
            "gemini_model": {
                "mean_score": float(gemini_mean_score),
                "std_score": float(gemini_std_score),
                "decision_distribution": dict(gemini_dist)
            },
            "comparison": {
                "score_diff": float(gemini_mean_score - vector_mean_score),
                "agreement_rate": float(agreement_rate)
            }
        }
    
    def print_results(self, results: Dict):
        """In kết quả đánh giá"""
        print("\n" + "=" * 80)
        print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH DECISION")
        print("=" * 80)
        
        print(f"\n📊 Số lượng mẫu đánh giá: {results['n_samples']}")
        print(f"📊 Mô hình được đánh giá:")
        if results.get('enable_vector'):
            print(f"  ✓ Vector (ML) model")
        if results.get('enable_llm'):
            print(f"  ✓ Gemini (LLM) model")
        
        if "ground_truths" in results:
            # Có ground truth
            metrics = results["metrics"]
            
            print("\n" + "-" * 80)
            print("METRICS VỚI GROUND TRUTH")
            print("-" * 80)
            
            if results.get('enable_vector') and 'vector_model' in metrics:
                print("\n🔵 VECTOR (ML) MODEL:")
                print(f"  Accuracy:  {metrics['vector_model']['accuracy']:.4f}")
                print(f"  Precision: {metrics['vector_model']['precision']:.4f}")
                print(f"  Recall:    {metrics['vector_model']['recall']:.4f}")
                print(f"  F1-Score:  {metrics['vector_model']['f1_score']:.4f}")
                
                # Hiển thị per-class metrics
                if 'classification_report' in metrics['vector_model']:
                    report = metrics['vector_model']['classification_report']
                    print("\n  Per-class metrics:")
                    labels = ["treat", "review", "test"]
                    for label in labels:
                        if label in report:
                            prec = report[label].get('precision', 0)
                            rec = report[label].get('recall', 0)
                            f1 = report[label].get('f1-score', 0)
                            support = report[label].get('support', 0)
                            print(f"    {label:6}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Support={support}")
                
                print("\n📊 CONFUSION MATRIX - VECTOR MODEL:")
                print("        treat  review  test")
                cm = metrics['vector_model']['confusion_matrix']
                labels = ["treat", "review", "test"]
                for i, label in enumerate(labels):
                    print(f"{label:8} {cm[i]}")
            
            if results.get('enable_llm') and 'gemini_model' in metrics:
                print("\n🟢 GEMINI (LLM) MODEL:")
                print(f"  Accuracy:  {metrics['gemini_model']['accuracy']:.4f}")
                print(f"  Precision: {metrics['gemini_model']['precision']:.4f}")
                print(f"  Recall:    {metrics['gemini_model']['recall']:.4f}")
                print(f"  F1-Score:  {metrics['gemini_model']['f1_score']:.4f}")
                
                # Hiển thị per-class metrics
                if 'classification_report' in metrics['gemini_model']:
                    report = metrics['gemini_model']['classification_report']
                    print("\n  Per-class metrics:")
                    labels = ["treat", "review", "test"]
                    for label in labels:
                        if label in report:
                            prec = report[label].get('precision', 0)
                            rec = report[label].get('recall', 0)
                            f1 = report[label].get('f1-score', 0)
                            support = report[label].get('support', 0)
                            print(f"    {label:6}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Support={support}")
                
                print("\n📊 CONFUSION MATRIX - GEMINI MODEL:")
                print("        treat  review  test")
                cm = metrics['gemini_model']['confusion_matrix']
                labels = ["treat", "review", "test"]
                for i, label in enumerate(labels):
                    print(f"{label:8} {cm[i]}")
            
            if 'comparison' in metrics:
                print("\n📈 SO SÁNH:")
                comp = metrics["comparison"]
                print(f"  Accuracy diff:  {comp['accuracy_diff']:+.4f} ({comp['winner_accuracy'].upper()} tốt hơn)")
                print(f"  Precision diff: {comp['precision_diff']:+.4f}")
                print(f"  Recall diff:    {comp['recall_diff']:+.4f}")
                print(f"  F1-Score diff:  {comp['f1_diff']:+.4f} ({comp['winner_f1'].upper()} tốt hơn)")
        else:
            # Không có ground truth - chỉ so sánh
            metrics = results["metrics"]
            
            print("\n" + "-" * 80)
            print("SO SÁNH TRỰC TIẾP 2 MÔ HÌNH")
            print("-" * 80)
            
            print(f"\n📊 Tỷ lệ đồng ý: {metrics['agreement_rate']:.4f}")
            
            print("\n🔵 VECTOR (ML) MODEL:")
            print(f"  Mean Score: {metrics['vector_model']['mean_score']:.4f}")
            print(f"  Std Score:  {metrics['vector_model']['std_score']:.4f}")
            print(f"  Distribution: {metrics['vector_model']['decision_distribution']}")
            
            print("\n🟢 GEMINI (LLM) MODEL:")
            print(f"  Mean Score: {metrics['gemini_model']['mean_score']:.4f}")
            print(f"  Std Score:  {metrics['gemini_model']['std_score']:.4f}")
            print(f"  Distribution: {metrics['gemini_model']['decision_distribution']}")
            
            print(f"\n📈 Score difference: {metrics['comparison']['score_diff']:+.4f}")
        
        # Phân tích methods được sử dụng
        print("\n" + "-" * 80)
        print("PHƯƠNG PHÁP ĐƯỢC SỬ DỤNG")
        print("-" * 80)
        
        from collections import Counter
        
        if results.get('enable_vector') and len(results['vector_methods']) > 0:
            vector_methods_count = Counter(results['vector_methods'])
            print("\n🔵 VECTOR MODEL methods:")
            for method, count in vector_methods_count.items():
                print(f"  {method}: {count} ({count/len(results['vector_methods'])*100:.1f}%)")
        
        if results.get('enable_llm') and len(results['gemini_methods']) > 0:
            gemini_methods_count = Counter(results['gemini_methods'])
            print("\n🟢 GEMINI MODEL methods:")
            for method, count in gemini_methods_count.items():
                print(f"  {method}: {count} ({count/len(results['gemini_methods'])*100:.1f}%)")
    
    def save_results(self, results: Dict, output_path: str = "logs/decision_evaluation.json"):
        """Lưu kết quả vào file JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Thêm timestamp
        results_with_meta = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✓ Đã lưu kết quả vào {output_path}")


def main():
    """Ví dụ sử dụng hệ thống đánh giá"""
    print("=" * 80)
    print("KHỞI TẠO HỆ THỐNG ĐÁNH GIÁ")
    print("=" * 80)
    
    # Load hoặc train hệ thống
    system = MASClinicalDecisionSystem()
    
    csv_path = "data/Bacteria_dataset_Multiresictance.csv"
    if not os.path.exists(csv_path):
        csv_path = "Bacteria_dataset_Multiresictance.csv"
    
    if not os.path.exists(csv_path):
        print("❌ Không tìm thấy file dữ liệu.")
        return
    
    # Kiểm tra xem đã train chưa
    if not system.is_trained:
        print("  ⚠️  Hệ thống chưa được train. Đang train...")
        system.train(csv_path, test_size=0.2, random_state=42)
    else:
        print("  ✓ Hệ thống đã được train. Đang load...")
        try:
            system.load()
        except:
            print("  ⚠️  Không thể load. Đang train lại...")
            system.train(csv_path, test_size=0.2, random_state=42)
    
    # Chọn mô hình để đánh giá (có thể enable cả 2 hoặc chỉ 1)
    enable_vector = True   # Set False để tắt Vector model
    enable_llm = True      # Set False để tắt Gemini LLM model
    
    # Lấy Gemini API key từ environment variable (tự động load từ .env nếu có)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  CẢNH BÁO: Không tìm thấy GEMINI_API_KEY trong environment variables!")
        print("   Vui lòng đặt API key bằng một trong các cách sau:")
        print("   1. Tạo file .env với nội dung: GEMINI_API_KEY=your-api-key-here (KHUYẾN NGHỊ)")
        print("   2. Windows PowerShell: $env:GEMINI_API_KEY='your-api-key-here'")
        print("   3. Windows CMD: set GEMINI_API_KEY=your-api-key-here")
        print("   4. Linux/Mac: export GEMINI_API_KEY='your-api-key-here'")
        print("\n   Lấy API key mới tại: https://aistudio.google.com/app/apikey")
        print("   ⚠️  KHÔNG BAO GIỜ hardcode API key trong code!")
        print("   📖 Xem hướng dẫn chi tiết trong file ENV_SETUP.md")
        if enable_llm:
            print("\n   ⚠️  LLM sẽ không hoạt động nếu không có API key hợp lệ.")
            response = input("\n   Bạn có muốn tiếp tục mà không dùng LLM? (y/n): ")
            if response.lower() != 'y':
                print("   Đang dừng...")
                return
            enable_llm = False
    
    # Tạo evaluator
    evaluator = DecisionModelEvaluator(
        system, 
        gemini_api_key=gemini_api_key,
        enable_vector=enable_vector,
        enable_llm=enable_llm
    )
    
    # Đánh giá trên dataset (có thể giới hạn số mẫu để test nhanh)
    print("\n" + "=" * 80)
    print("BẮT ĐẦU ĐÁNH GIÁ")
    print("=" * 80)
    
    # Thời gian delay giữa các lần gọi LLM (giây) để tránh quota
    # Đặt 0 để không delay, hoặc 15 để delay 15 giây giữa mỗi mẫu
    llm_delay = 15.0  # Delay 15 giây giữa mỗi lần gọi LLM
    
    results = evaluator.evaluate_on_dataset(
        csv_path,
        n_samples=20,  # Giới hạn 50 mẫu để test nhanh, có thể tăng hoặc để None
        use_ground_truth=True,
        llm_delay=llm_delay  # Delay giữa các lần gọi LLM
    )
    
    # In kết quả
    evaluator.print_results(results)
    
    # Lưu kết quả
    evaluator.save_results(results)


if __name__ == "__main__":
    main()

