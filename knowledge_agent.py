import sqlite3
import os
from typing import Dict, Any
from llama_engine import LlamaEngine

class KnowledgeAgent:
    def __init__(self, db_path: str = "data/knowledge_base.db"):
        self.db_path = db_path
        self.engine = LlamaEngine()

    def query_merchant(self, entity_name: str) -> Dict[str, Any]:
        if not os.path.exists(self.db_path):
            return {}
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT category, amenities FROM merchants WHERE name LIKE ?", (f"%{entity_name}%",))
        row = cur.fetchone()
        conn.close()
        if row:
            return {"category": row[0], "amenities": [a.strip() for a in row[1].split(",")]}
        return {}

    def verify_factuality(self, text: str) -> Dict[str, Any]:
        prompt = f"Extract entity_name and claimed_amenity from text: '{text}'. Return JSON."
        claims = self.engine.extract_json(prompt)
        
        entity = claims.get("entity_name")
        claimed_amenity = claims.get("claimed_amenity")
        
        if not entity or not claimed_amenity:
            return {"hallucination_detected": False, "reason": "No explicit factual claim detected"}
        
        record = self.query_merchant(entity)
        if not record:
            return {"hallucination_detected": False, "reason": f"Entity '{entity}' not in registry"}
        
        amenities = record.get("amenities", [])
        if claimed_amenity.lower() not in [a.lower() for a in amenities]:
            return {
                "hallucination_detected": True,
                "reason": f"Claimed amenity '{claimed_amenity}' not in registered amenities for '{entity}' ({', '.join(amenities)})"
            }
        
        return {"hallucination_detected": False, "reason": "Claim verified against knowledge base"}
