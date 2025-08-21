import os
import json
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn
from typing import Optional
from torchvision import models


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str) -> torch.Tensor:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(image).unsqueeze(0)  # 1x3x224x224
    return tensor


def load_model(model_path: str, num_classes: int = 2, device: Optional[torch.device] = None) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = models.resnet18(weights=None)
    except Exception:
        model = models.resnet18(pretrained=False)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_class_mapping(mapping_path: str) -> dict:
    with open(mapping_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(idx): cls for cls, idx in class_to_idx.items()}
    return idx_to_class


def predict(image_path: str) -> None:
    model_path = os.path.join("model", "trained_model.pth")
    mapping_path = os.path.join("model", "class_to_idx.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping not found at {mapping_path}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=2, device=device)
    idx_to_class = load_class_mapping(mapping_path)

    input_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    pred_class = idx_to_class.get(pred_idx, str(pred_idx))
    print(f"Prediction: {pred_class}  |  Confidence: {confidence*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Real/Fake from a frame image")
    parser.add_argument("--image", "-i", type=str, help="Path to image file")
    args = parser.parse_args()

    if args.image:
        predict(args.image)
    else:
        path = input("Enter test image path (e.g. extracted_frames/real/real_xxx.jpg): ").strip()
        predict(path)
