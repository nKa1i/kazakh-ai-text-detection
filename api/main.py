from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# We import model.py which will load the model at startup
# and fail fast if the model isn't available.
try:
    import model
except SystemExit:
    # Rethrow so the app truly exits if model import exits
    raise
except Exception as e:
    import sys
    print(f"Failed to initialize model module: {e}")
    sys.exit(1)

app = FastAPI(title="KazRoBERTa AI-Text Detector API")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to analyze")

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
    return model.predict(request.text)

@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    result = model.predict(request.text)
    result["html"] = model.explain(request.text)
    return result