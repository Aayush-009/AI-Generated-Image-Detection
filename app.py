from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import onnxruntime as ort
import numpy as np
from huggingface_hub import hf_hub_download
import os

app = Flask(__name__)
CORS(app)


# ========================================
# DOWNLOAD ONNX MODEL FROM HUGGING FACE
# ========================================

model_path = hf_hub_download(
    repo_id="Aayush-009/ai-image-detector-model",
    filename="ai_detector.onnx"
)

# Download external ONNX data file
hf_hub_download(
    repo_id="Aayush-009/ai-image-detector-model",
    filename="ai_detector.onnx.data"
)


# ========================================
# LOAD ONNX MODEL
# ========================================

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX model loaded successfully.")


# ========================================
# IMAGE PREPROCESSING
# ========================================

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


# ========================================
# HOME PAGE
# ========================================

@app.route("/")
def home():

    return render_template("index.html")


# ========================================
# PREDICTION
# ========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "No image selected."
        }), 400

    try:

        # Open image
        image = Image.open(file).convert("RGB")

        # Preprocess
        image = preprocess_image(image)

        # Run ONNX prediction
        output = session.run(
            [output_name],
            {
                input_name: image
            }
        )[0]

        # Softmax
        exp = np.exp(
            output - np.max(output)
        )

        probabilities = (
            exp /
            exp.sum(
                axis=1,
                keepdims=True
            )
        )

        # Class mapping
        classes = [
            "FAKE",
            "REAL"
        ]

        predicted_index = np.argmax(
            probabilities
        )

        prediction = classes[
            predicted_index
        ]

        confidence = float(
            probabilities[0][predicted_index]
            * 100
        )

        return jsonify({
            "prediction": prediction,
            "confidence": round(
                confidence,
                2
            )
        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ========================================
# START SERVER
# ========================================

if __name__ == "__main__":

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
