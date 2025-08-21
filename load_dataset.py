import os
import cv2
import numpy as np

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    return image.flatten() / 255.0  # Normalize pixel values

def load_dataset():
    data = []
    labels = []

    for label_dir, label_value in [("frames/fake", 0), ("frames/real", 1)]:
        for filename in os.listdir(label_dir):
            if filename.endswith(".jpg"):
                path = os.path.join(label_dir, filename)
                features = extract_features(path)
                data.append(features)
                labels.append(label_value)

    return np.array(data), np.array(labels)

if __name__ == "__main__":
    X, y = load_dataset()
    print(f"✅ Loaded dataset: {X.shape[0]} samples, each with {X.shape[1]} features")
