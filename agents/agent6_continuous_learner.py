"""
Agent 6 - Continuous Learner
Nhiệm vụ: Học từ dữ liệu mới và cập nhật mô hình theo thời gian
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import os
import json
try:
    from agents.agent3_resistance_predictor import ResistancePredictorAgent
except ImportError:
    # Fallback nếu import không thành công
    ResistancePredictorAgent = None


class ContinuousLearnerAgent:
    def __init__(self, model_path: str = "models/", 
                 performance_log_path: str = "logs/performance_log.json"):
        self.model_path = model_path
        self.performance_log_path = performance_log_path
        self.performance_history = []
        self.threshold_for_retrain = 0.02  # Retrain nếu accuracy giảm > 2%
        
        # Tạo thư mục nếu chưa có
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(os.path.dirname(performance_log_path) if os.path.dirname(performance_log_path) else '.', exist_ok=True)
        
        # Load performance history nếu có
        self._load_performance_history()
    
    def monitor_performance(self,
                           model,
                           X_test: pd.DataFrame,
                           y_test: pd.DataFrame,
                           test_name: str = "validation") -> Dict:
        """
        Theo dõi hiệu suất mô hình
        """
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # Dự đoán
        y_pred, y_proba = model.predict(X_test)
        
        # Tính metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Tính ROC-AUC trung bình
        auc_scores = []
        for idx, col in enumerate(y_test.columns):
            try:
                auc = roc_auc_score(
                    y_test.iloc[:, idx],
                    y_proba.iloc[:, idx]
                )
                auc_scores.append(auc)
            except:
                pass
        
        avg_auc = np.mean(auc_scores) if auc_scores else 0.0
        
        # Lưu kết quả
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'accuracy': float(accuracy),
            'avg_auc': float(avg_auc),
            'num_samples': len(X_test),
            'num_features': X_test.shape[1]
        }
        
        self.performance_history.append(performance_record)
        self._save_performance_history()
        
        return performance_record
    
    def check_retrain_needed(self, current_accuracy: float) -> bool:
        """
        Kiểm tra xem có cần huấn luyện lại không
        """
        if len(self.performance_history) < 2:
            return False
        
        # So sánh với accuracy trước đó
        previous_accuracies = [p['accuracy'] for p in self.performance_history[-5:]]
        if previous_accuracies:
            avg_previous = np.mean(previous_accuracies[:-1])
            
            # Nếu accuracy giảm đáng kể
            if current_accuracy < avg_previous - self.threshold_for_retrain:
                return True
        
        return False
    
    def incremental_update(self,
                          model,
                          X_new: pd.DataFrame,
                          y_new: pd.DataFrame,
                          retrain_full: bool = False):
        """
        Cập nhật mô hình với dữ liệu mới
        """
        if retrain_full:
            # Huấn luyện lại từ đầu với toàn bộ dữ liệu
            print("Đang huấn luyện lại mô hình với toàn bộ dữ liệu...")
            # Note: Cần có X_full và y_full để retrain
            # Đây là placeholder, cần implement logic đầy đủ
            return model
        else:
            # Incremental learning (nếu mô hình hỗ trợ)
            # Với Random Forest/XGBoost, thường cần retrain với dữ liệu mới
            print(f"Nhận {len(X_new)} mẫu dữ liệu mới.")
            print("Lưu ý: XGBoost/RandomForest cần retrain với toàn bộ dữ liệu.")
            
            return model
    
    def schedule_retrain(self,
                       model,
                       X_all: pd.DataFrame,
                       y_all: pd.DataFrame,
                       schedule_type: str = "weekly") -> Dict:
        """
        Lên lịch huấn luyện lại định kỳ
        """
        timestamp = datetime.now()
        
        # Kiểm tra xem đã đến lúc retrain chưa
        if schedule_type == "weekly":
            # Retrain mỗi tuần
            last_retrain = self._get_last_retrain_date()
            if last_retrain:
                days_since = (timestamp - last_retrain).days
                if days_since < 7:
                    return {
                        'should_retrain': False,
                        'days_until_retrain': 7 - days_since
                    }
        
        # Thực hiện retrain
        print(f"Bắt đầu huấn luyện lại mô hình theo lịch ({schedule_type})...")
        
        # Lưu mô hình cũ
        old_model_path = os.path.join(self.model_path, f"model_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}.pkl")
        model.save_model(old_model_path)
        
        # Tạo mô hình mới và train
        if ResistancePredictorAgent is None:
            raise ImportError("Không thể import ResistancePredictorAgent")
        new_model = ResistancePredictorAgent(model_type=model.model_type)
        results = new_model.train(X_all, y_all)
        
        # Lưu mô hình mới
        new_model_path = os.path.join(self.model_path, "model_latest.pkl")
        new_model.save_model(new_model_path)
        
        # Lưu thông tin retrain
        retrain_record = {
            'timestamp': timestamp.isoformat(),
            'schedule_type': schedule_type,
            'model_path': new_model_path,
            'backup_path': old_model_path,
            'performance': results
        }
        
        self.performance_history.append({
            'type': 'retrain',
            **retrain_record
        })
        self._save_performance_history()
        
        return {
            'should_retrain': True,
            'retrained': True,
            'results': results
        }
    
    def _get_last_retrain_date(self) -> Optional[datetime]:
        """Lấy ngày retrain cuối cùng"""
        for record in reversed(self.performance_history):
            if record.get('type') == 'retrain':
                return datetime.fromisoformat(record['timestamp'])
        return None
    
    def get_performance_trend(self, metric: str = 'accuracy') -> Dict:
        """
        Phân tích xu hướng hiệu suất
        """
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        metrics = [p.get(metric, 0) for p in self.performance_history if metric in p]
        
        if len(metrics) < 2:
            return {'trend': 'insufficient_data'}
        
        # Tính xu hướng
        recent_avg = np.mean(metrics[-5:]) if len(metrics) >= 5 else np.mean(metrics[-len(metrics):])
        older_avg = np.mean(metrics[:-5]) if len(metrics) >= 10 else np.mean(metrics[:len(metrics)//2])
        
        if recent_avg > older_avg:
            trend = 'improving'
        elif recent_avg < older_avg:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_average': float(recent_avg),
            'older_average': float(older_avg),
            'change': float(recent_avg - older_avg)
        }
    
    def _load_performance_history(self):
        """Tải lịch sử hiệu suất"""
        if os.path.exists(self.performance_log_path):
            try:
                with open(self.performance_log_path, 'r', encoding='utf-8') as f:
                    self.performance_history = json.load(f)
            except:
                self.performance_history = []
        else:
            self.performance_history = []
    
    def _save_performance_history(self):
        """Lưu lịch sử hiệu suất"""
        try:
            with open(self.performance_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Lỗi khi lưu performance history: {e}")
    
    def export_performance_report(self, output_path: str = "reports/performance_report.txt"):
        """
        Xuất báo cáo hiệu suất
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        report = "=" * 80 + "\n"
        report += "BÁO CÁO HIỆU SUẤT MÔ HÌNH\n"
        report += "=" * 80 + "\n\n"
        
        # Xu hướng
        trend = self.get_performance_trend('accuracy')
        report += f"Xu hướng hiệu suất: {trend.get('trend', 'unknown')}\n"
        if 'change' in trend:
            report += f"Thay đổi: {trend['change']:.4f}\n"
        report += "\n"
        
        # Lịch sử gần đây
        report += "Lịch sử hiệu suất (5 lần gần nhất):\n"
        report += "-" * 80 + "\n"
        for record in self.performance_history[-5:]:
            if 'accuracy' in record:
                report += f"{record.get('timestamp', 'N/A')}: "
                report += f"Accuracy = {record['accuracy']:.4f}\n"
        
        # Lưu file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Báo cáo đã được xuất tại: {output_path}")

