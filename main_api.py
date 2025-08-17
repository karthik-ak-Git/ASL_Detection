import io
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from config import (FRONTEND_DIR, OUTPUTS_DIR, MODEL_PATH,
                    CLASS_TO_IDX_PATH, IDX_TO_CLASS_PATH,
                    API_TITLE, API_VERSION)

app = FastAPI(title=API_TITLE, version=API_VERSION)

# CORS: if serving frontend from same origin, you can tighten this further.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to specific origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class TopKItem(BaseModel):
    label: str
    confidence: float


class Prediction(BaseModel):
    label: str
    confidence: float
    top5: List[TopKItem]


# Load model & mapping at startup
NUM_CLASSES = 29
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()

# Load mappings
with open(IDX_TO_CLASS_PATH, "r") as f:
    idx_to_class = json.load(f)  # keys may be strings if saved from dict
# normalize keys to int
idx_to_class = {int(k): v for k, v in idx_to_class.items()}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/labels")
def labels():
    return {"labels": [idx_to_class[i] for i in range(len(idx_to_class))]}


@app.post("/api/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    with torch.no_grad():
        x = preprocess(image).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top5_prob, top5_idx = torch.topk(probs, 5)

    top5 = [
        {"label": idx_to_class[int(i)], "confidence": float(p)}
        for p, i in zip(top5_prob.tolist(), top5_idx.tolist())
    ]
    return {"label": top5[0]["label"], "confidence": top5[0]["confidence"], "top5": top5}

# Serve frontend AFTER API routes are defined
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR),
          html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
