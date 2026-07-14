from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# We import model.py which will load the model at startup
# and fail fast if the model isn't available.
try:
    try:
        from api import model
    except ImportError:
        import model
except Exception as e:
    print(f"Model initialization skipped/failed: {e}")
    model = None

app = FastAPI(title="KazRoBERTa AI-Text Detector API")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to analyze")
    mode: str = Field("pure", description="The mode to run: pure or fst")

class PredictResponse(BaseModel):
    label: str
    confidence: float

class ExplainResponse(BaseModel):
    label: str
    confidence: float
    html: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    return model.predict(request.text, request.mode)

@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    result = model.predict(request.text, request.mode)
    result["html"] = model.explain(request.text, request.mode)
    return result

from typing import Dict, Any, Optional
from consensus_engine import ConsensusEngine

consensus_engine = ConsensusEngine()

class VerificationRequest(BaseModel):
    text: str
    telemetry: Optional[Dict[str, Any]] = None

@app.post("/v1/verify")
def verify_content(req: VerificationRequest):
    result = consensus_engine.evaluate(req.text, req.telemetry, layer1_score=0.10)
    return result
