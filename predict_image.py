import argparse
import io
import json
import os
import sys
import urllib.request
import logging
import csv

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from utils.logger import setup_logger
from config import MODEL_PATH, IDX_TO_CLASS_PATH, OUTPUTS_DIR, LOGS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = setup_logger("predict_image", logfile=LOGS_DIR / "predict_image.log")


def load_idx_to_class():
    with open(IDX_TO_CLASS_PATH, "r") as f:
        idx_to_class = json.load(f)
    return {int(k): v for k, v in idx_to_class.items()}


def load_model(model_path: str, num_classes: int):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    # Use the same transforms as the pretrained weights
    weights = ResNet18_Weights.DEFAULT
    return weights.transforms()


def load_image_from_source(src: str) -> Image.Image:
    if src.lower().startswith("http://") or src.lower().startswith("https://"):
        # Some hosts block default Python UA; set a browser-like UA and timeout
        req = urllib.request.Request(
            src,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    else:
        if not os.path.exists(src):
            raise FileNotFoundError(f"File not found: {src}")
        return Image.open(src).convert("RGB")


def predict_one(model, img: Image.Image, idx_to_class):
    transform = get_transform()
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        top5_prob, top5_idx = probs.topk(5)
    top5 = [(idx_to_class[i.item()], top5_prob[j].item())
            for j, i in enumerate(top5_idx)]
    pred_label, pred_conf = top5[0]
    return pred_label, pred_conf, top5


def main():
    parser = argparse.ArgumentParser(
        description="Predict ASL class for an image path or URL.")
    parser.add_argument("source", help="Path to image or direct image URL")
    args = parser.parse_args()

    # Ensure directories exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(
            f"Model weights not found at {MODEL_PATH}. Train the model first.")
        sys.exit(1)

    print(f"Using device: {DEVICE}")
    LOGGER.info(f"Predicting on device={DEVICE}")
    idx_to_class = load_idx_to_class()
    model = load_model(str(MODEL_PATH), num_classes=len(idx_to_class))

    try:
        img = load_image_from_source(args.source)
    except Exception as e:
        print(f"Failed to load image: {e}")
        sys.exit(2)

    label, conf, top5 = predict_one(model, img, idx_to_class)
    print(f"Prediction: {label} (confidence: {conf:.3f})")
    print("Top-5:")
    for cls, p in top5:
        print(f"  - {cls}: {p:.3f}")
    # Persist logs and a simple CSV append
    try:
        csv_path = OUTPUTS_DIR / "predictions_single.csv"
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["source", "prediction", "confidence"])
            w.writerow([args.source, label, f"{conf:.6f}"])
        LOGGER.info(f"Predicted {label} ({conf:.3f}) for {args.source}")
    except Exception as e:
        LOGGER.error(f"Failed to write prediction CSV/log: {e}")


if __name__ == "__main__":
    main()
