from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
from huggingface_hub import hf_hub_downloadfrom flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import onnxruntime as ort
import numpy as np
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# Download ONNX model files from Hugging Face
onnx_model = hf_hub_download(
    repo_id="Aayush-009/ai-image-detector-model",
    filename="ai_detector.onnx"
)

onnx_data = hf_hub_download(
    repo_id="Aayush-009/ai-image-detector-model",
    filename="ai_detector.onnx.data"
)

# ONNX model
session = ort.InferenceSession(
    onnx_model,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX model loaded successfully.")


# Image preprocessing
def preprocess_image(image):

    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    try:

        image = Image.open(file).convert("RGB")

        image = preprocess_image(image)

        output = session.run(
            [output_name],
            {input_name: image}
        )[0]

        # Softmax
        exp = np.exp(
            output - np.max(output)
        )

        probabilities = (
            exp /
            exp.sum(axis=1, keepdims=True)
        )

        classes = ["FAKE", "REAL"]

        predicted = np.argmax(probabilities)

        prediction = classes[predicted]

        confidence = float(
            probabilities[0][predicted] * 100
        )

        return jsonify({
            "prediction": prediction,
            "confidence": round(
                confidence, 2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    import os

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )

app = Flask(__name__)
CORS(app)

# Use CPU on Render
device = torch.device("cpu")

print("Using device:", device)


# ========================================
# DOWNLOAD MODEL FROM HUGGING FACE
# ========================================

model_path = hf_hub_download(
    repo_id="Aayush-009/ai-image-detector-model",
    filename="ai_detector.pth"
)


# ========================================
# LOAD RESNET18
# ========================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)


model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ========================================
# IMAGE PREPROCESSING
# ========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ========================================
# HOME PAGE
# ========================================

@app.route("/")
def home():

    return render_template("index.html")


# ========================================
# PREDICTION API
# ========================================

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


        # CPU

        image = image.to(device)


        # Prediction

        with torch.inference_mode():

            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            confidence, predicted = torch.max(
                probabilities,
                1
            )


        # Classes

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

        print("Prediction error:", e)

        return jsonify({

            "error": str(e)

        }), 500


# ========================================
# RUN APP
# ========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=10000
    )
