# KazRoBERTa AI-Text Detector — Deployment Design

**Date:** 2026-05-09  
**Project:** kazakh-ai-text-detection  
**Goal:** Deploy the trained KazRoBERTa Pure classifier as a FastAPI + Gradio app in Docker Compose, suitable as a portfolio demo.

---

## Architecture

Two services managed by Docker Compose:

- **`api`** — FastAPI app. Loads KazRoBERTa Pure at startup. Exposes `/health` and `/predict` endpoints.
- **`ui`** — Gradio app. Thin frontend that calls `http://api:8000/predict` over the internal Docker network.

Both services share an internal Docker network (`app-network`). The UI container never loads the model — it is purely a frontend making HTTP requests.

```
User → Gradio UI (localhost:7860)
         → POST http://api:8000/predict
         → KazRoBERTa inference
         → {"label": "ai"|"human", "confidence": float}
         → Gradio displays result
```

---

## File Structure

```
D:/roberta/
├── api/
│   ├── main.py          # FastAPI app, routes
│   ├── model.py         # Model load + inference
│   ├── requirements.txt
│   └── Dockerfile
├── ui/
│   ├── app.py           # Gradio interface
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md            # Updated with demo instructions
```

---

## Components

### `api/model.py`
- Loads tokenizer + model from `data/pure_KazRoBERTa/` at module import time
- `predict(text: str) -> dict` — tokenizes with `truncation=True, max_length=512`, runs inference, returns `{"label": "human"|"ai", "confidence": float}`
- Model loaded once at startup, reused across requests

### `api/main.py`
- `GET /health` → `{"status": "ok"}`
- `POST /predict` → body: `{"text": str}` → response: `{"label": str, "confidence": float}`
- Validates that `text` is non-empty (422 if missing)
- Model load failure at startup → process exits with error log (fail fast)

### `ui/app.py`
- Single Gradio `Interface`: text input → calls `POST http://api:8000/predict` → displays label + confidence
- Empty/short input (<10 chars) → shows warning, skips API call
- API unreachable → catches `requests.exceptions.ConnectionError`, shows "Service unavailable"

### `docker-compose.yml`
- `api` service: build `./api`, port `8000:8000`, volume mount `./data/pure_KazRoBERTa:/model` (read-only)
- `ui` service: build `./ui`, port `7860:7860`, depends_on `api`
- Shared network: `app-network`

### Dockerfiles
- `api/Dockerfile`: Python 3.11-slim, installs torch (CPU) + transformers + fastapi + uvicorn
- `ui/Dockerfile`: Python 3.11-slim, installs gradio + requests

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Empty or <10 char input | Gradio warns client-side, no API call |
| Input >512 tokens | Tokenizer truncates automatically |
| API unreachable | Gradio catches exception, shows "Service unavailable" |
| Model load failure | FastAPI fails fast at startup with error log |

---

## Testing

- `api/test_api.py` — two pytest tests:
  - `test_health`: GET `/health` → 200, `{"status": "ok"}`
  - `test_predict`: POST `/predict` with a Kazakh text sample → label in `["human", "ai"]`, confidence in `[0, 1]`
- Manual smoke test: `docker compose up` → open `localhost:7860` → paste sample text → verify prediction

No unit tests for Gradio UI — integration via smoke test is sufficient for a thin wrapper.

---

## Model Choice

**KazRoBERTa Pure** (`data/pure_KazRoBERTa/`) — 96.10% accuracy on `heldout_test.csv`. Best overall accuracy. KazRoBERTa FST (lowest FPs, 36) is an alternative if precision is prioritized.

---

## Out of Scope

- Authentication / rate limiting
- GPU inference (CPU is sufficient for demo traffic)
- CI/CD pipeline
- Cloud deployment (Hugging Face Spaces or similar can be added later)
