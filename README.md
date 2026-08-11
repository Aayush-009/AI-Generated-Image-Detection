# AI Image Detector

An AI-powered web application that detects whether an uploaded image is **AI-generated or Real** using a **ResNet18 Convolutional Neural Network (CNN)**.

## Overview

The AI Image Detector uses deep learning to classify images into two categories:

- **FAKE** – AI-generated image
- **REAL** – Real image

The trained ResNet18 model is integrated with a Flask backend and a modern web interface built using HTML, CSS, and JavaScript.

## Features

- AI-generated image detection
- Real image detection
- JPG, JPEG, and PNG image support
- Drag-and-drop image upload
- Image preview
- Confidence score
- Confidence progress bar
- Loading indicator
- Reset and Analyze Another Image options
- CUDA GPU acceleration
- Responsive web interface

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Computer vision and ResNet18 |
| ResNet18 | Image classification model |
| Flask | Backend web framework |
| HTML | Web structure |
| CSS | User interface styling |
| JavaScript | Frontend functionality |
| Pillow | Image processing |
| CUDA | GPU acceleration |

## Model

**Architecture:** ResNet18  
**Task:** Binary Image Classification  
**Classes:** FAKE and REAL  
**Input Size:** 224 × 224 pixels

The final classification layer of ResNet18 is modified to classify the two target classes.

## Application Workflow

```text
User Uploads Image
        ↓
Frontend
        ↓
Flask API
        ↓
Image Preprocessing
        ↓
ResNet18 Model
        ↓
FAKE / REAL Prediction
        ↓
Confidence Score
        ↓
Result Displayed

AI Image Detector/
│
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── README.md
│
├── model/
│   └── ai_detector.pth
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js

Installation
1. Clone or Download the Project

Download the repository and open the project folder in VS Code.

2. Create a Virtual Environment
python -m venv .venv
3. Activate the Virtual Environment

For Windows:

.venv\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
How to Run

Start the Flask application:

python app.py

The application will run locally at:

http://127.0.0.1:5000

Open the address in a web browser.

Using the Application
Upload an image using Choose Image or drag and drop.
Preview the selected image.
Click Analyze Image.
The ResNet18 model processes the image.
The application displays the predicted class and confidence score.
Click Analyze Another Image to test another image.
GPU Acceleration

The application supports CUDA-enabled GPU acceleration when a compatible NVIDIA GPU and CUDA-enabled PyTorch installation are available.

Development hardware:

NVIDIA GeForce RTX 3050 Laptop GPU
CUDA
Requirements

The required Python packages are listed in:

requirements.txt

A virtual environment such as .venv should not be included when sharing or uploading the project. Other users can create their own environment and install the required packages using requirements.txt.

Important

The trained model file is required for prediction. Make sure the model file is present at the expected location before running the application.

Future Improvements
Improve model accuracy
Add support for more image formats
Add prediction history
Add detailed analysis reports
Deploy the application online
Experiment with additional deep learning architectures
Disclaimer

This application provides predictions based on a trained machine learning model. The result should not be considered absolute proof that an image is AI-generated or real.

License

This project is intended for educational and academic purposes.
