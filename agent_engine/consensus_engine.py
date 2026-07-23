from typing import Dict, Any, Optional
from telemetry_guard import TelemetryGuard
from knowledge_agent import KnowledgeAgent

class ConsensusEngine:
    """
    Consensus Engine aggregating Layer 1 (KazRoBERTa), Layer 2 (TelemetryGuard),
    and Layer 3 (KnowledgeAgent) into a weighted content risk evaluation.
    """
    def __init__(self, db_path: str = "data/knowledge_base.db"):
        self.telemetry_guard = TelemetryGuard()
        self.knowledge_agent = KnowledgeAgent(db_path=db_path)

    def evaluate(
        self,
        text: str,
        telemetry: Optional[Dict[str, Any]] = None,
        layer1_score: float = 0.10
    ) -> Dict[str, Any]:
        telemetry = telemetry or {}
        
        l2_res = self.telemetry_guard.evaluate(telemetry)
        l2_score = l2_res.get("telemetry_risk_score", 0.0)
        l2_flags = l2_res.get("flags", [])
        
        l3_res = self.knowledge_agent.verify_factuality(text)
        hallucination_detected = l3_res.get("hallucination_detected", False)
        l3_flag = 1.0 if hallucination_detected else 0.0
        
        # Weighted overall risk calculation
        overall_score = (0.40 * layer1_score) + (0.35 * l2_score) + (0.25 * l3_flag)
        overall_score = round(min(1.0, max(0.0, overall_score)), 2)
        
        if overall_score > 0.70:
            action = "REJECT"
        elif overall_score > 0.35:
            action = "MODERATE"
        else:
            action = "APPROVE"
            
        all_flags = list(l2_flags)
        if l3_flag > 0:
            all_flags.append("HALLUCINATION_CONTRADICTION")
            
        return {
            "is_authentic": action == "APPROVE",
            "overall_risk_score": overall_score,
            "action_recommended": action,
            "breakdown": {
                "layer1_linguistic_score": layer1_score,
                "layer2_telemetry_score": l2_score,
                "layer3_hallucination_detected": hallucination_detected,
                "flags_triggered": all_flags
            }
        }
