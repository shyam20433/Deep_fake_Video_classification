import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   
                         [0.229, 0.224, 0.225])   
])

train_data = datasets.ImageFolder("extracted_frames", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
except Exception:
    model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 4 
num_batches = len(train_loader)
total_steps = max(epochs * max(num_batches, 1), 1)

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        steps_done = epoch * max(num_batches, 1) + (batch_index + 1)
        percent_complete = 100.0 * steps_done / total_steps
        print(f"Progress: {percent_complete:.2f}% ({steps_done}/{total_steps})", end="\r")

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
          f"Accuracy: {100*correct/total:.2f}%")

print(f"Progress: 100.00% ({total_steps}/{total_steps})")

os.makedirs("model", exist_ok=True)

model_path = os.path.join("model", "trained_model.pth")
torch.save(model.state_dict(), model_path)

class_to_idx_path = os.path.join("model", "class_to_idx.json")
with open(class_to_idx_path, "w", encoding="utf-8") as f:
    json.dump(train_data.class_to_idx, f, indent=2)

print(f"Saved model to {model_path}")
print(f"Saved class mapping to {class_to_idx_path}")
