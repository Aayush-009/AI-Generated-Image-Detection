```python
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# Download Model from Hugging Face
# ==========================================

model_path = hf_hub_download(
    repo_id="Aayush--009/ai-image-detector-model",
    filename="ai_detector.pth"
)


# ==========================================
# Load ResNet18
# ==========================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ==========================================
# Image Preprocessing
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Prediction API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    try:

        # Open image
        image = Image.open(file).convert("RGB")

        # Preprocess
        image = transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        # Move to device
        image = image.to(device)

        # Prediction
        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        # Get prediction
        confidence, predicted = torch.max(
            probabilities,
            1
        )

        # Class names
        classes = [
            "FAKE",
            "REAL"
        ]

        prediction = classes[
            predicted.item()
        ]

        confidence = (
            confidence.item() * 100
        )

        return jsonify({
            "prediction": prediction,
            "confidence": round(
                confidence,
                2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# Run Flask Application
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True
    )
```
