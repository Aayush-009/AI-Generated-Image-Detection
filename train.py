import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Using Device:", device)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(10),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
)])

train_dataset = datasets.ImageFolder("deepfake_dataset/train", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Classes:", train_dataset.classes)
print("Class Mapping:", train_dataset.class_to_idx)
print("Total images trained:", len(train_dataset))

model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 2)
print("Model Architecture:", model)

# Move model to RTX 3050
model = model.to(device)

print("ResNet18 loaded successfully.")

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Move data to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# ==========================================
# 8. SAVE MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

torch.save(
    model.state_dict(),
    "model/ai_detector.pth"
)

print()
print("===================================")
print("Training completed successfully!")
print("Model saved: model/ai_detector.pth")
print("===================================")
