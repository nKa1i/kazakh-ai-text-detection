import json
import requests
from typing import Dict, Any

class LlamaEngine:
    def __init__(self, api_url: str = "http://localhost:8080/v1/chat/completions"):
        self.api_url = api_url

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            r = requests.post(self.api_url, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # Fallback for offline/test environments
            return f"{{\"error\": \"LLM unreachable: {str(e)}\"}}"

    def extract_json(self, prompt: str) -> Dict[str, Any]:
        raw = self.generate(prompt)
        try:
            # Extract JSON substring if surrounded by markdown code blocks
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_text": raw, "parse_error": True}
