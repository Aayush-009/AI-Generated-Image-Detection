import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Pytorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Using Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Test Image Transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(10),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
)])

# Load Test Dataset

test_dataset = datasets.ImageFolder("deepfake_dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print("Classes:", test_dataset.classes)
print("Class Mapping:", test_dataset.class_to_idx)
print("Total images tested:", len(test_dataset))

# Load the trained model ResNet18

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model/ai_detector.pth", map_location=device))

model = model.to(device)
model.eval()

print("Model loaded successfully.")

# Test Model

all_labels = []
all_predictions = []

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Test Accuracy

accuracy = 100 * correct / total
print()
print("="*50)
print(f"Test Accuracy: {accuracy:.2f}%")
print("="*50)   

# Classification Report

print()
print("Classification Report:")
print()
print(classification_report(all_labels, all_predictions, target_names=test_dataset.classes))

# Confusion Matrix

print("Confusion Matrix:")

cm = confusion_matrix(all_labels, all_predictions)
print(cm)
print()
print("Evaluation completed successfully!")
