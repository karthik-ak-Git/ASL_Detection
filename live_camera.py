import random
import cv2
import torch
import json
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
from config import MODEL_PATH, IDX_TO_CLASS_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class mappings
with open(IDX_TO_CLASS_PATH, "r") as f:
    idx_to_class = json.load(f)
idx_to_class = {int(k): v for k, v in idx_to_class.items()}

# Assign a unique color for each letter
random.seed(42)


def get_color(label):
    # Map label to a unique color
    colors = {}
    for i, l in enumerate(idx_to_class.values()):
        # Use HSV to RGB for visually distinct colors
        hsv = np.array([i / len(idx_to_class), 1, 1])
        rgb = tuple(int(x * 255) for x in cv2.cvtColor(
            np.uint8([[hsv * [179, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]/255)
        colors[l] = rgb
    return colors.get(label, (0, 255, 0))


# Load model with current TorchVision API
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Transform: use the same transforms as the pretrained weights
transform = weights.transforms()

# ✅ Use DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Define center crop region (simulate hand region)
    h, w, _ = frame.shape
    crop_size = min(h, w, 224)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - crop_size // 2, cy - crop_size // 2
    x2, y2 = x1 + crop_size, y1 + crop_size
    # Clamp to frame
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]

    # Preprocess
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        label = idx_to_class[predicted.item()]

    # Draw circle and label
    color = get_color(label)
    # Draw circle at center of crop
    cv2.circle(frame, (cx, cy), crop_size//2, color, 3)
    # Draw label above circle
    text = f"{label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
    text_x = cx - tw // 2
    text_y = cy - crop_size//2 - 10
    text_y = max(th+10, text_y)
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)

    cv2.imshow("ASL Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
