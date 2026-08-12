import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load("model/ai_detector.pth", map_location=device)
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")

# Image Transformation
# IMPORTANT: No RandomFlip or RandomRotation during prediction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Image Path
image_path = input("Enter Image Path: ")

# Load Image
image = Image.open(image_path).convert("RGB")

image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

# Prediction
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)

    fake_probability = probabilities[0][0].item()
    real_probability = probabilities[0][1].item()

# Result
if fake_probability > real_probability:
    prediction = "AI GENERATED"
    confidence = fake_probability * 100
else:
    prediction = "REAL"
    confidence = real_probability * 100

print()
print("====================================================")
print("              AI IMAGE DETECTION")
print("====================================================")
print("Prediction:", prediction)
print(f"Confidence: {confidence:.2f}%")
print(f"AI Generated Probability: {fake_probability * 100:.2f}%")
print(f"Real Probability: {real_probability * 100:.2f}%")
print("====================================================")
