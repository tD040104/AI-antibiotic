"""
Decision agent - combine critic insights + treatment recommendations
Enhanced with Fuzzy Logic, Decision Tree, and LLM for decision making
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False
    fuzz = None  # type: ignore
    ctrl = None  # type: ignore

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None  # type: ignore

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None  # type: ignore


class FuzzyDecisionSystem:
    """Fuzzy logic system for decision making."""
    
    def __init__(self):
        if not SKFUZZY_AVAILABLE:
            self.available = False
            return
        
        self.available = True
        
        # Define fuzzy variables
        # Probability: 0 to 1
        self.probability = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'probability')
        self.probability['low'] = fuzz.trimf(self.probability.universe, [0, 0, 0.4])
        self.probability['medium'] = fuzz.trimf(self.probability.universe, [0.3, 0.5, 0.7])
        self.probability['high'] = fuzz.trimf(self.probability.universe, [0.6, 1, 1])
        
        # Uncertainty: 0 to 1
        self.uncertainty = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'uncertainty')
        self.uncertainty['low'] = fuzz.trimf(self.uncertainty.universe, [0, 0, 0.3])
        self.uncertainty['medium'] = fuzz.trimf(self.uncertainty.universe, [0.2, 0.5, 0.8])
        self.uncertainty['high'] = fuzz.trimf(self.uncertainty.universe, [0.7, 1, 1])
        
        # Risk factors: 0 to 5
        self.risk_factors = ctrl.Antecedent(np.arange(0, 6, 1), 'risk_factors')
        self.risk_factors['low'] = fuzz.trimf(self.risk_factors.universe, [0, 0, 2])
        self.risk_factors['medium'] = fuzz.trimf(self.risk_factors.universe, [1, 2.5, 4])
        self.risk_factors['high'] = fuzz.trimf(self.risk_factors.universe, [3, 5, 5])
        
        # Decision score: 0 to 1
        self.decision_score = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'decision_score')
        self.decision_score['treat'] = fuzz.trimf(self.decision_score.universe, [0.7, 1, 1])
        self.decision_score['review'] = fuzz.trimf(self.decision_score.universe, [0.3, 0.5, 0.7])
        self.decision_score['test'] = fuzz.trimf(self.decision_score.universe, [0, 0, 0.4])
        
        # Define rules
        rule1 = ctrl.Rule(
            self.probability['high'] & self.uncertainty['low'] & self.risk_factors['low'],
            self.decision_score['treat']
        )
        rule2 = ctrl.Rule(
            self.probability['high'] & self.uncertainty['low'] & self.risk_factors['medium'],
            self.decision_score['treat']
        )
        rule3 = ctrl.Rule(
            self.probability['medium'] | self.uncertainty['medium'],
            self.decision_score['review']
        )
        rule4 = ctrl.Rule(
            self.probability['low'] | self.uncertainty['high'] | self.risk_factors['high'],
            self.decision_score['test']
        )
        
        self.control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)
    
    def compute_decision(
        self, 
        probability: float, 
        uncertainty: float, 
        risk_factors: float
    ) -> Dict:
        """Compute fuzzy decision."""
        if not self.available:
            return {
                "decision_type": "fuzzy_unavailable",
                "decision_score": 0.5,
                "confidence": "medium"
            }
        
        try:
            self.simulator.input['probability'] = max(0, min(1, probability))
            self.simulator.input['uncertainty'] = max(0, min(1, uncertainty))
            self.simulator.input['risk_factors'] = max(0, min(5, risk_factors))
            
            self.simulator.compute()
            
            score = self.simulator.output['decision_score']
            
            if score >= 0.7:
                decision_type = "treat"
                confidence = "high"
            elif score >= 0.4:
                decision_type = "review"
                confidence = "medium"
            else:
                decision_type = "test"
                confidence = "low"
            
            return {
                "decision_type": decision_type,
                "decision_score": float(score),
                "confidence": confidence,
                "method": "fuzzy_logic"
            }
        except Exception as e:
            return {
                "decision_type": "fuzzy_error",
                "decision_score": 0.5,
                "confidence": "medium",
                "error": str(e)
            }


class DecisionTreeRecommender:
    """Decision tree for treatment recommendation."""
    
    def __init__(self):
        self.tree = DecisionTreeRegressor(max_depth=5, random_state=42)
        self.is_trained = False
        self.label_encoder = LabelEncoder()
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train decision tree."""
        self.tree.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> float:
        """Predict using decision tree."""
        if not self.is_trained:
            return 0.5  # Default score
        
        prediction = self.tree.predict(X.reshape(1, -1))
        return float(prediction[0])


class LLMDecisionMaker:
    """LLM-based decision maker using Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        # Sử dụng Gemini thay vì OpenAI
        self.available = GEMINI_AVAILABLE and self.api_key is not None
        
        if self.available:
            genai.configure(api_key=self.api_key)
            # Tự động tìm model hợp lệ nếu model mặc định không tồn tại
            self.model = self._find_valid_model(model)
    
    def _find_valid_model(self, preferred_model: str) -> str:
        """Tự động tìm model hợp lệ từ danh sách các model phổ biến."""
        # Danh sách các model để thử (theo thứ tự ưu tiên)
        models_to_try = [
            preferred_model,  # Thử model được chỉ định trước
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro",
        ]
        
        # Thử list models từ API
        try:
            available_models = genai.list_models()
            model_names = []
            for m in available_models:
                # Kiểm tra xem model có hỗ trợ generateContent không
                if hasattr(m, 'supported_generation_methods'):
                    if 'generateContent' in m.supported_generation_methods:
                        # Lấy tên model, có thể là full name hoặc short name
                        model_name = getattr(m, 'name', '')
                        if model_name:
                            model_names.append(model_name)
                elif hasattr(m, 'name'):
                    # Nếu không có supported_generation_methods, thử dùng luôn
                    model_names.append(m.name)
            
            # Ưu tiên model trong danh sách models_to_try
            for model_name in models_to_try:
                # Kiểm tra cả với và không có prefix "models/"
                for full_name in [model_name, f"models/{model_name}"]:
                    if full_name in model_names:
                        print(f"  ✓ Đã tìm thấy model hợp lệ: {full_name}")
                        return full_name
                    # Kiểm tra nếu model name kết thúc bằng tên model (loại bỏ prefix)
                    for available_name in model_names:
                        # Loại bỏ prefix "models/" nếu có
                        clean_name = available_name.replace("models/", "")
                        if clean_name == model_name or available_name.endswith(model_name):
                            print(f"  ✓ Đã tìm thấy model hợp lệ: {available_name}")
                            return available_name
            
            # Nếu không tìm thấy trong danh sách ưu tiên, lấy model đầu tiên có sẵn
            if model_names:
                print(f"  ⚠️  Sử dụng model mặc định từ API: {model_names[0]}")
                return model_names[0]
        except Exception as e:
            # Không in lỗi ở đây để tránh spam, chỉ log khi cần
            pass
        
        # Fallback: thử từng model trong danh sách bằng cách test thực tế
        for model_name in models_to_try:
            try:
                test_model = genai.GenerativeModel(model_name)
                # Thử generate một test ngắn để xác nhận model hoạt động
                test_response = test_model.generate_content(
                    "test", 
                    generation_config=genai.types.GenerationConfig(max_output_tokens=1)
                )
                print(f"  ✓ Đã tìm thấy model hợp lệ qua test: {model_name}")
                return model_name
            except Exception:
                continue
        
        # Nếu không tìm thấy model nào, trả về model mặc định và để lỗi xảy ra khi sử dụng
        print(f"  ⚠️  Không tìm thấy model hợp lệ, sẽ sử dụng: {preferred_model} (có thể gây lỗi khi sử dụng)")
        return preferred_model
    
    def generate_decision(
        self,
        patient_data: Dict,
        probabilities: Dict,
        critic_report: Dict,
        explanation: Dict
    ) -> Dict:
        """Generate decision using LLM."""
        if not self.available:
            return {
                "decision_type": "llm_unavailable",
                "decision_score": 0.5,
                "confidence": "medium"
            }
        
        try:
            prompt = self._create_prompt(patient_data, probabilities, critic_report, explanation)
            
            # Create full prompt with system message
            full_prompt = f"You are a medical AI assistant helping with antibiotic resistance decisions.\n\n{prompt}"
            
            # Use Gemini API
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.3
                )
            )
            
            # Handle response properly - check multiple ways to extract text
            if hasattr(response, 'text') and response.text:
                decision_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Try to extract text from candidates
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    decision_text = ''.join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                else:
                    decision_text = str(response)
            else:
                decision_text = str(response)
            
            # Parse decision from LLM response
            decision = self._parse_llm_response(decision_text)
            decision["method"] = "llm_gemini"
            decision["llm_response"] = decision_text[:200] if len(decision_text) > 200 else decision_text  # Truncate for storage
            
            return decision
        except Exception as e:
            # Log error for debugging
            error_msg = str(e)
            # Check if it's a model not found error - thử tự động tìm model hợp lệ
            if "404" in error_msg or "not found" in error_msg.lower():
                # Thử tự động tìm model hợp lệ một lần nữa
                old_model = self.model
                self.model = self._find_valid_model(self.model)
                
                # Nếu tìm thấy model mới, thử lại
                if self.model != old_model:
                    try:
                        prompt = self._create_prompt(patient_data, probabilities, critic_report, explanation)
                        full_prompt = f"You are a medical AI assistant helping with antibiotic resistance decisions.\n\n{prompt}"
                        model = genai.GenerativeModel(self.model)
                        response = model.generate_content(
                            full_prompt,
                            generation_config=genai.types.GenerationConfig(
                                max_output_tokens=200,
                                temperature=0.3
                            )
                        )
                        
                        # Handle response properly
                        if hasattr(response, 'text') and response.text:
                            decision_text = response.text
                        elif hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                decision_text = ''.join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                            else:
                                decision_text = str(response)
                        else:
                            decision_text = str(response)
                        
                        decision = self._parse_llm_response(decision_text)
                        decision["method"] = "llm_gemini"
                        decision["llm_response"] = decision_text[:200] if len(decision_text) > 200 else decision_text
                        return decision
                    except Exception as retry_error:
                        error_msg = f"Model '{old_model}' not found, đã thử model '{self.model}' nhưng vẫn lỗi: {str(retry_error)}"
                else:
                    error_msg = f"Model '{self.model}' not found. Không tìm thấy model hợp lệ. Original error: {error_msg}"
            # Don't print here to avoid spam, but include in return
            return {
                "decision_type": "llm_error",
                "decision_score": 0.5,
                "confidence": "medium",
                "error": error_msg,
                "method": "llm_gemini_error"
            }
    
    def _create_prompt(
        self,
        patient_data: Dict,
        probabilities: Dict,
        critic_report: Dict,
        explanation: Dict
    ) -> str:
        """Create prompt for LLM."""
        prompt = f"""
Based on the following clinical data, provide a treatment decision:

Patient: {patient_data.get('age', 'Unknown')} years, {patient_data.get('gender', 'Unknown')}
Bacteria: {patient_data.get('bacteria', 'Unknown')}
Risk factors: Diabetes={patient_data.get('diabetes', 'No')}, Hypertension={patient_data.get('hypertension', 'No')}

Top antibiotic probabilities:
"""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        for ab, prob in sorted_probs:
            prompt += f"  - {ab}: {prob:.2%}\n"
        
        prompt += f"\nUncertainty: {critic_report.get('decision', {}).get('uncertainty_score', 0):.2f}\n"
        prompt += "\nProvide decision: TREAT, REVIEW, or TEST. Explain briefly."
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response to extract decision."""
        response_lower = response.lower()
        
        if "treat" in response_lower and "test" not in response_lower:
            decision_type = "treat"
            decision_score = 0.8
            confidence = "high"
        elif "test" in response_lower or "additional" in response_lower:
            decision_type = "test"
            decision_score = 0.3
            confidence = "low"
        else:
            decision_type = "review"
            decision_score = 0.5
            confidence = "medium"
        
        return {
            "decision_type": decision_type,
            "decision_score": decision_score,
            "confidence": confidence
        }


class TreatmentRecommenderAgent:
    """Domain rules to rank antibiotics based on sensitivity probability.
    Enhanced with fuzzy logic and decision tree."""

    def __init__(self, use_fuzzy: bool = True, use_decision_tree: bool = True):
        self.antibiotic_names = {
            "AMX/AMP": "Amoxicillin/Ampicillin",
            "AMC": "Amoxicillin-Clavulanic Acid",
            "CZ": "Cefazolin",
            "FOX": "Cefoxitin",
            "CTX/CRO": "Ceftriaxone/Cefotaxime",
            "IPM": "Imipenem",
            "GEN": "Gentamicin",
            "AN": "Amikacin",
            "Acide nalidixique": "Nalidixic Acid",
            "ofx": "Ofloxacin",
            "CIP": "Ciprofloxacin",
            "C": "Chloramphenicol",
            "Co-trimoxazole": "Trimethoprim-Sulfamethoxazole",
            "Furanes": "Nitrofurantoin",
            "colistine": "Colistin",
        }

        self.antibiotic_groups = {
            "Beta-lactams": ["AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO"],
            "Carbapenems": ["IPM"],
            "Aminoglycosides": ["GEN", "AN"],
            "Quinolones": ["Acide nalidixique", "ofx", "CIP"],
            "Others": ["C", "Co-trimoxazole", "Furanes", "colistine"],
        }

        self.antibiotic_priority = {
            "IPM": 1,
            "AN": 2,
            "GEN": 3,
            "CIP": 4,
            "ofx": 5,
            "CTX/CRO": 6,
            "AMC": 7,
            "CZ": 8,
            "FOX": 9,
            "AMX/AMP": 10,
            "colistine": 1,
            "Co-trimoxazole": 11,
            "Acide nalidixique": 12,
            "C": 13,
            "Furanes": 14,
        }
        
        # Initialize fuzzy system and decision tree
        self.fuzzy_system = FuzzyDecisionSystem() if use_fuzzy else None
        self.decision_tree = DecisionTreeRecommender() if use_decision_tree else None

    def recommend_treatment(
        self,
        resistance_proba: pd.DataFrame,
        patient_data: Optional[pd.Series] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        recommendations: List[Dict] = []
        if len(resistance_proba) == 0:
            return recommendations
        proba = resistance_proba.iloc[0]
        sorted_antibiotics = proba.sort_values(ascending=False)
        sensitive_antibiotics = sorted_antibiotics[sorted_antibiotics >= 0.5]
        if len(sensitive_antibiotics) == 0:
            sensitive_antibiotics = sorted_antibiotics.head(top_k)

        # Use fuzzy logic or decision tree instead of if-else
        recommendations = self._apply_fuzzy_medical_guidelines(
            sensitive_antibiotics, patient_data, top_k
        )
        return recommendations

    def _apply_fuzzy_medical_guidelines(
        self,
        sensitive_proba: pd.Series,
        patient_data: Optional[pd.Series] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        """Apply fuzzy logic and decision tree instead of if-else."""
        candidates = []
        risk_factors = patient_data.get("Total_risk_factors", 0) if patient_data is not None else 0
        
        for antibiotic, proba in sensitive_proba.items():
            # Use fuzzy system to compute score
            if self.fuzzy_system and self.fuzzy_system.available:
                uncertainty = 1.0 - proba  # Higher uncertainty for lower probability
                fuzzy_result = self.fuzzy_system.compute_decision(
                    probability=proba,
                    uncertainty=uncertainty,
                    risk_factors=float(risk_factors)
                )
                base_score = fuzzy_result.get("decision_score", proba)
            else:
                base_score = proba
            
            # Apply priority using fuzzy membership
            priority_score = base_score
            if antibiotic in self.antibiotic_priority:
                priority = self.antibiotic_priority[antibiotic]
                # Fuzzy priority adjustment: higher priority = higher bonus
                priority_bonus = 1.0 / (priority + 1) * 0.2
                priority_score = base_score + priority_bonus
            
            # Use decision tree if available
            if self.decision_tree and self.decision_tree.is_trained:
                feature_vector = np.array([
                    proba,
                    float(risk_factors),
                    float(self.antibiotic_priority.get(antibiotic, 10)),
                    float(patient_data.get("Age", 50)) if patient_data is not None else 50.0
                ])
                tree_score = self.decision_tree.predict(feature_vector)
                # Combine fuzzy and tree scores
                final_score = (priority_score * 0.6 + tree_score * 0.4)
            else:
                final_score = priority_score

            candidates.append({
                "antibiotic": antibiotic,
                "full_name": self.antibiotic_names.get(antibiotic, antibiotic),
                "sensitive_probability": proba,
                "score": final_score,
                "group": self._get_antibiotic_group(antibiotic),
            })

        # Sort by score (fuzzy/decision tree based)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top k using fuzzy grouping
        selected = []
        groups_used = set()
        for candidate in candidates:
            if len(selected) >= top_k:
                break
            # Use fuzzy membership for group selection
            if candidate["group"] not in groups_used or len(groups_used) >= top_k:
                selected.append(candidate)
                groups_used.add(candidate["group"])

        # Fill remaining slots
        while len(selected) < top_k and len(selected) < len(candidates):
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                    break

        recommendations = []
        for idx, candidate in enumerate(selected[:top_k]):
            recommendations.append({
                "rank": idx + 1,
                "antibiotic_code": candidate["antibiotic"],
                "antibiotic_name": candidate["full_name"],
                "sensitive_probability": round(candidate["sensitive_probability"], 3),
                "confidence": self._get_confidence_level(candidate["sensitive_probability"]),
                "group": candidate["group"],
                "recommendation": self._generate_recommendation_text(candidate),
            })
        return recommendations

    def _get_antibiotic_group(self, antibiotic: str) -> str:
        for group, antibiotics in self.antibiotic_groups.items():
            if antibiotic in antibiotics:
                return group
        return "Others"

    def _get_confidence_level(self, proba: float) -> str:
        if proba >= 0.8:
            return "High"
        if proba >= 0.6:
            return "Medium"
        return "Low"

    def _generate_recommendation_text(self, candidate: Dict) -> str:
        proba = candidate["sensitive_probability"]
        if proba >= 0.8:
            return "Highly recommended - High sensitivity probability"
        if proba >= 0.6:
            return "Recommended - Moderate sensitivity probability"
        return "Consider with caution - Lower sensitivity probability"


class DecisionAgent:
    """
    Agent 5: combine critic feedback + treatment ranking to output actions.
    Enhanced with fuzzy logic, decision tree, and LLM.
    Now accepts vector inputs from Agent 3 and Agent 4 for direct ML model usage.
    """

    def __init__(
        self, 
        treatment_agent: Optional[TreatmentRecommenderAgent] = None,
        use_fuzzy: bool = True,
        use_decision_tree: bool = True,
        use_llm: bool = False,
        llm_api_key: Optional[str] = None,
        decision_mode: str = "llm_only"  # "vector_only", "llm_only", "auto" (use both, prefer LLM if available)
    ):
        self.treatment_agent = treatment_agent or TreatmentRecommenderAgent(
            use_fuzzy=use_fuzzy,
            use_decision_tree=use_decision_tree
        )
        self.llm_decision_maker = None
        self.decision_mode = decision_mode  # "vector_only", "llm_only", "auto"
        
        if use_llm and decision_mode != "vector_only":
            self.llm_decision_maker = LLMDecisionMaker(api_key=llm_api_key)
        
        # Decision tree trained on combined vectors from Agent 3 and Agent 4
        self.combined_decision_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
        self.is_tree_trained = False

    def decide(
        self,
        probabilities: pd.DataFrame,
        critic_report: Dict,
        patient_features: Optional[pd.Series] = None,
        explanation: Optional[Dict] = None,
        explain_vector: Optional[np.ndarray] = None,
        critic_vector: Optional[np.ndarray] = None,
        top_k: int = 3,
    ) -> Dict:
        """Make decision using fuzzy logic, decision tree, and optionally LLM.
        
        Args:
            probabilities: Prediction probabilities
            critic_report: Critic agent report (dict format, for backward compatibility)
            patient_features: Patient feature data
            explanation: Explanation dict (for backward compatibility)
            explain_vector: Vector from Agent 3 (ExplainabilityAgent) - preferred input
            critic_vector: Vector from Agent 4 (CriticAgent) - preferred input
            top_k: Number of top recommendations
        """
        recommendations = self.treatment_agent.recommend_treatment(
            probabilities,
            patient_features,
            top_k=top_k,
        )

        # Chọn luồng xử lý dựa trên decision_mode
        if self.decision_mode == "llm_only":
            # Chỉ dùng LLM, không dùng vector
            explain_decision = explanation.get("decision", {}) if explanation else {}
            critic_decision = critic_report.get("decision", {})
            final_decision = self._combine_decisions(
                explain_decision,
                critic_decision,
                recommendations,
                probabilities,
                patient_features
            )
        elif self.decision_mode == "vector_only":
            # Chỉ dùng Vector (ML), không dùng LLM
            if explain_vector is not None and critic_vector is not None:
                final_decision = self._decide_from_vectors(
                    explain_vector,
                    critic_vector,
                    probabilities,
                    patient_features,
                    recommendations
                )
            else:
                # Fallback nếu không có vectors
                explain_decision = explanation.get("decision", {}) if explanation else {}
                critic_decision = critic_report.get("decision", {})
                final_decision = self._combine_decisions(
                    explain_decision,
                    critic_decision,
                    recommendations,
                    probabilities,
                    patient_features
                )
        else:  # "auto" mode - ưu tiên LLM nếu có, fallback về vector
            if explain_vector is not None and critic_vector is not None:
                # Use vector inputs directly for ML models
                final_decision = self._decide_from_vectors(
                    explain_vector,
                    critic_vector,
                    probabilities,
                    patient_features,
                    recommendations
                )
            else:
                # Fallback to dict-based decision (backward compatibility)
                explain_decision = explanation.get("decision", {}) if explanation else {}
                critic_decision = critic_report.get("decision", {})
                final_decision = self._combine_decisions(
                    explain_decision,
                    critic_decision,
                    recommendations,
                    probabilities,
                    patient_features
                )

        # Use fuzzy logic for action determination instead of if-else
        actions = self._generate_fuzzy_actions(
            critic_report, 
            recommendations, 
            probabilities,
            patient_features
        )

        return {
            "primary_actions": actions,
            "therapy_recommendations": recommendations,
            "decision": final_decision,  # Main decision output
        }
    
    def _decide_from_vectors(
        self,
        explain_vector: np.ndarray,
        critic_vector: np.ndarray,
        probabilities: pd.DataFrame,
        patient_features: Optional[pd.Series],
        recommendations: List[Dict]
    ) -> Dict:
        """Make decision directly from Agent 3 and Agent 4 vectors using ML models."""
        # If LLM is available and enabled, use it first (even with vectors)
        # Nhưng chỉ khi decision_mode không phải "vector_only"
        if (self.decision_mode != "vector_only" and 
            self.llm_decision_maker and 
            self.llm_decision_maker.available):
            try:
                patient_dict = patient_features.to_dict() if patient_features is not None else {}
                prob_dict = probabilities.iloc[0].to_dict() if not probabilities.empty else {}
                critic_decision = {"uncertainty_score": float(np.mean(critic_vector[:5])) if len(critic_vector) >= 5 else 0.5}
                explain_decision = {"decision_score": float(np.mean(explain_vector)) if len(explain_vector) > 0 else 0.5}
                
                llm_decision = self.llm_decision_maker.generate_decision(
                    patient_data=patient_dict,
                    probabilities=prob_dict,
                    critic_report={"decision": critic_decision},
                    explanation={"decision": explain_decision}
                )
                
                # Kiểm tra xem LLM có trả về lỗi không
                decision_type = llm_decision.get("decision_type", "")
                error_msg = llm_decision.get("error", "")
                
                # Nếu LLM trả về lỗi do API key (403), disable LLM và fallback
                if decision_type == "llm_error" and ("403" in error_msg or "leaked" in error_msg.lower() or "API key" in error_msg):
                    print(f"  ⚠️  LLM API key không hợp lệ hoặc bị leak. Đang fallback về Vector (ML) model.")
                    self.llm_decision_maker.available = False
                    # Fall through to ML models
                elif decision_type not in ["llm_unavailable", "llm_error"]:
                    return {
                        "decision_type": decision_type,
                        "decision_score": llm_decision.get("decision_score", 0.5),
                        "confidence": llm_decision.get("confidence", "medium"),
                        "method": llm_decision.get("method", "llm_gemini_on_vectors"),
                        "reasoning": "Gemini LLM decision using vector-enhanced context",
                        "recommended_antibiotic": recommendations[0]["antibiotic_code"] if recommendations else None
                    }
                # Nếu LLM trả về lỗi khác, fall through to ML models
            except Exception as e:
                error_str = str(e)
                # Nếu là lỗi API key, disable LLM
                if "403" in error_str or "leaked" in error_str.lower() or "API key" in error_str:
                    print(f"  ⚠️  LLM API key không hợp lệ. Đang fallback về Vector (ML) model.")
                    if self.llm_decision_maker:
                        self.llm_decision_maker.available = False
                # Fall through to ML models if LLM fails
                pass
        
        # Combine vectors from Agent 3 and Agent 4
        combined_vector = np.concatenate([explain_vector, critic_vector])
        
        # Use Decision Tree if trained
        if self.is_tree_trained:
            try:
                decision_score = self.combined_decision_tree.predict(
                    combined_vector.reshape(1, -1)
                )[0]
                decision_score = float(np.clip(decision_score, 0.0, 1.0))
                
                if decision_score >= 0.7:
                    decision_type = "treat"
                    confidence = "high"
                elif decision_score >= 0.4:
                    decision_type = "review"
                    confidence = "medium"
                else:
                    decision_type = "test"
                    confidence = "low"
                
                return {
                    "decision_type": decision_type,
                    "decision_score": decision_score,
                    "confidence": confidence,
                    "method": "decision_tree_on_vectors",
                    "reasoning": "Decision tree trained on Agent 3 and Agent 4 vectors",
                    "recommended_antibiotic": recommendations[0]["antibiotic_code"] if recommendations else None
                }
            except Exception:
                pass  # Fall through to fuzzy logic
        
        # Use Fuzzy Logic with vector inputs (chỉ khi không phải llm_only mode)
        if (self.decision_mode != "llm_only" and 
            self.treatment_agent.fuzzy_system and 
            self.treatment_agent.fuzzy_system.available):
            # Extract key features from combined vector
            mean_prob = probabilities.iloc[0].mean() if not probabilities.empty else 0.5
            # Use first few elements of explain_vector as uncertainty proxy
            uncertainty = float(np.mean(critic_vector[:5])) if len(critic_vector) >= 5 else 0.5
            risk_factors = patient_features.get("Total_risk_factors", 0) if patient_features is not None else 0
            
            fuzzy_result = self.treatment_agent.fuzzy_system.compute_decision(
                probability=mean_prob,
                uncertainty=uncertainty,
                risk_factors=float(risk_factors)
            )
            
            return {
                "decision_type": fuzzy_result.get("decision_type", "review"),
                "decision_score": fuzzy_result.get("decision_score", 0.5),
                "confidence": fuzzy_result.get("confidence", "medium"),
                "method": "fuzzy_logic_on_vectors",
                "reasoning": "Fuzzy logic applied to Agent 3 and Agent 4 vectors",
                "recommended_antibiotic": recommendations[0]["antibiotic_code"] if recommendations else None
            }
        
        # Fallback: simple weighted combination of vector means
        explain_mean = float(np.mean(explain_vector))
        critic_mean = float(np.mean(critic_vector))
        combined_score = (explain_mean * 0.6 + (1 - critic_mean) * 0.4)
        
        if combined_score >= 0.7:
            decision_type = "treat"
            confidence = "high"
        elif combined_score >= 0.4:
            decision_type = "review"
            confidence = "medium"
        else:
            decision_type = "test"
            confidence = "low"
        
        return {
            "decision_type": decision_type,
            "decision_score": float(combined_score),
            "confidence": confidence,
            "method": "vector_weighted_average",
            "reasoning": "Weighted average of Agent 3 and Agent 4 vector means",
            "recommended_antibiotic": recommendations[0]["antibiotic_code"] if recommendations else None
        }
    
    def train_decision_tree(
        self,
        explain_vectors: np.ndarray,
        critic_vectors: np.ndarray,
        decision_scores: np.ndarray
    ):
        """Train decision tree on combined vectors from Agent 3 and Agent 4.
        
        Args:
            explain_vectors: Array of shape (n_samples, explain_dim) from Agent 3
            critic_vectors: Array of shape (n_samples, critic_dim) from Agent 4
            decision_scores: Array of shape (n_samples,) with target decision scores
        """
        # Combine vectors
        combined_vectors = np.concatenate([explain_vectors, critic_vectors], axis=1)
        
        # Train decision tree
        self.combined_decision_tree.fit(combined_vectors, decision_scores)
        self.is_tree_trained = True
    
    def _generate_fuzzy_actions(
        self,
        critic_report: Dict,
        recommendations: List[Dict],
        probabilities: pd.DataFrame,
        patient_features: Optional[pd.Series]
    ) -> List[str]:
        """Generate actions using fuzzy logic instead of if-else."""
        actions = []
        
        # Use fuzzy system to determine if additional tests are needed
        if self.treatment_agent.fuzzy_system and self.treatment_agent.fuzzy_system.available:
            uncertainty = critic_report.get("decision", {}).get("uncertainty_score", 0.5)
            mean_prob = probabilities.iloc[0].mean() if not probabilities.empty else 0.5
            risk_factors = patient_features.get("Total_risk_factors", 0) if patient_features is not None else 0
            
            fuzzy_result = self.treatment_agent.fuzzy_system.compute_decision(
                probability=mean_prob,
                uncertainty=uncertainty,
                risk_factors=float(risk_factors)
            )
            
            decision_type = fuzzy_result.get("decision_type", "review")
            
            # Generate actions based on fuzzy decision
            if decision_type == "test" or critic_report.get("needs_additional_tests"):
                if critic_report.get("flags"):
                    flagged_codes = [flag.antibiotic for flag in critic_report["flags"]]
                    actions.append(
                        f"Yêu cầu xét nghiệm bổ sung cho: {', '.join(flagged_codes)}"
                    )
                if critic_report.get("missing_fields"):
                    missing = ", ".join(critic_report["missing_fields"][:5])
                    actions.append(f"Bổ sung dữ liệu: {missing}")
            
            if decision_type == "treat" and recommendations:
                top_choice = recommendations[0]
                actions.append(
                    f"Xem xét sử dụng {top_choice['antibiotic_name']} "
                    f"(P(sensitive)={top_choice['sensitive_probability']:.2f})"
                )
            elif decision_type == "review":
                actions.append("Cần xem xét kỹ trước khi quyết định điều trị.")
            elif not recommendations:
                actions.append("Chưa có khuyến nghị điều trị rõ ràng, cần hội chẩn.")
        else:
            # Fallback to simple rules
            if critic_report.get("needs_additional_tests"):
                if critic_report.get("flags"):
                    flagged_codes = [flag.antibiotic for flag in critic_report["flags"]]
                    actions.append(
                        f"Yêu cầu xét nghiệm bổ sung cho: {', '.join(flagged_codes)}"
                    )
            
            if recommendations:
                top_choice = recommendations[0]
                actions.append(
                    f"Xem xét sử dụng {top_choice['antibiotic_name']} "
                    f"(P(sensitive)={top_choice['sensitive_probability']:.2f})"
                )
            else:
                actions.append("Chưa có khuyến nghị điều trị rõ ràng, cần hội chẩn.")
        
        return actions
    
    def _combine_decisions(
        self,
        explain_decision: Dict,
        critic_decision: Dict,
        recommendations: List[Dict],
        probabilities: pd.DataFrame,
        patient_features: Optional[pd.Series]
    ) -> Dict:
        """Combine decisions from multiple sources using fuzzy logic or LLM."""
        # Use LLM if available (chỉ khi không phải vector_only mode)
        if (self.decision_mode != "vector_only" and 
            self.llm_decision_maker and 
            self.llm_decision_maker.available):
            try:
                patient_dict = patient_features.to_dict() if patient_features is not None else {}
                prob_dict = probabilities.iloc[0].to_dict() if not probabilities.empty else {}
                
                llm_decision = self.llm_decision_maker.generate_decision(
                    patient_data=patient_dict,
                    probabilities=prob_dict,
                    critic_report={"decision": critic_decision},
                    explanation={"decision": explain_decision}
                )
                
                decision_type = llm_decision.get("decision_type", "")
                error_msg = llm_decision.get("error", "")
                
                # Nếu LLM trả về lỗi do API key (403), disable LLM và fallback
                if decision_type == "llm_error" and ("403" in error_msg or "leaked" in error_msg.lower() or "API key" in error_msg):
                    if self.llm_decision_maker:
                        self.llm_decision_maker.available = False
                    # Fall through to fuzzy logic
                elif decision_type not in ["llm_unavailable", "llm_error"]:
                    return {
                        "decision_type": decision_type,
                        "decision_score": llm_decision.get("decision_score", 0.5),
                        "confidence": llm_decision.get("confidence", "medium"),
                        "method": "llm",
                        "reasoning": llm_decision.get("reasoning", "LLM-based decision"),
                        "recommended_antibiotic": recommendations[0]["antibiotic_code"] if recommendations else None
                    }
            except Exception as e:
                error_str = str(e)
                # Nếu là lỗi API key, disable LLM
                if "403" in error_str or "leaked" in error_str.lower() or "API key" in error_str:
                    if self.llm_decision_maker:
                        self.llm_decision_maker.available = False
                # Fall through to fuzzy logic
                pass
        
        # Use fuzzy logic to combine decisions (chỉ khi không phải llm_only mode)
        if (self.decision_mode != "llm_only" and 
            self.treatment_agent.fuzzy_system and 
            self.treatment_agent.fuzzy_system.available):
            explain_score = explain_decision.get("decision_score", 0.5)
            critic_uncertainty = critic_decision.get("uncertainty_score", 0.5)
            mean_prob = probabilities.iloc[0].mean() if not probabilities.empty else 0.5
            risk_factors = patient_features.get("Total_risk_factors", 0) if patient_features is not None else 0
            
            fuzzy_result = self.treatment_agent.fuzzy_system.compute_decision(
                probability=mean_prob,
                uncertainty=critic_uncertainty,
                risk_factors=float(risk_factors)
            )
            
            return {
                "decision_type": fuzzy_result.get("decision_type", "review"),
                "decision_score": fuzzy_result.get("decision_score", 0.5),
                "confidence": fuzzy_result.get("confidence", "medium"),
                "method": "fuzzy_logic",
                "reasoning": "Fuzzy logic-based decision combining explainability and critic insights",
                "recommended_antibiotic": recommendations[0]["antibiotic_code"] if recommendations else None
            }
        
        # Fallback: combine scores
        explain_score = explain_decision.get("decision_score", 0.5)
        critic_score = 1.0 - critic_decision.get("uncertainty_score", 0.5)
        combined_score = (explain_score * 0.6 + critic_score * 0.4)
        
        if combined_score >= 0.7:
            decision_type = "treat"
            confidence = "high"
        elif combined_score >= 0.4:
            decision_type = "review"
            confidence = "medium"
        else:
            decision_type = "test"
            confidence = "low"
        
        return {
            "decision_type": decision_type,
            "decision_score": float(combined_score),
            "confidence": confidence,
            "method": "weighted_average",
            "reasoning": "Weighted combination of explainability and critic decisions",
            "recommended_antibiotic": recommendations[0]["antibiotic_code"] if recommendations else None
        }







