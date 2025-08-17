import argparse
import csv
import io
import os
import sys
import urllib.request
from typing import List
import logging

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from data.dataloader import get_class_mapping
from utils.logger import setup_logger

MODEL_PATH = "outputs/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = setup_logger(
    "predict_urls", logfile=os.path.join("logs", "predict_urls.log"))


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def fetch_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def predict_url(model, url: str, idx_to_class) -> tuple[str, float]:
    try:
        img = fetch_image(url)
        x = get_transform()(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            conf, idx = probs.max(dim=0)
            label = idx_to_class[idx.item()]
        return label, float(conf)
    except Exception as e:
        return f"ERROR: {e}", 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Predict ASL classes for a list of image URLs.")
    parser.add_argument(
        "--urls", help="Path to a text file containing one URL per line")
    parser.add_argument("--url", help="A single URL to predict", default=None)
    parser.add_argument("--out", help="Output CSV path",
                        default="outputs/predictions_urls.csv")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model weights not found at {MODEL_PATH}")
        sys.exit(1)

    class_to_idx, idx_to_class = get_class_mapping()
    model = load_model(num_classes=len(class_to_idx))

    urls: List[str] = []
    if args.urls:
        if not os.path.exists(args.urls):
            print(f"ERROR: URLs file not found: {args.urls}")
            sys.exit(2)
        with open(args.urls, "r", encoding="utf-8") as f:
            urls.extend([line.strip() for line in f if line.strip()])
        if not urls:
            print(f"ERROR: URLs file '{args.urls}' is empty.")
            sys.exit(2)
        print(f"Loaded {len(urls)} URLs from {args.urls}")
    if args.url:
        urls.append(args.url)

    if not urls:
        print("No URLs provided. Use --url or --urls <file>.")
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = []
    for u in urls:
        label, conf = predict_url(model, u, idx_to_class)
        results.append((u, label, conf))
        print(f"{u} -> {label} ({conf:.3f})")
        LOGGER.info(f"{u} -> {label} ({conf:.3f})")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "prediction", "confidence"])
        w.writerows(results)
    print(f"Wrote {len(results)} predictions to {args.out}")
    LOGGER.info(f"Wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
