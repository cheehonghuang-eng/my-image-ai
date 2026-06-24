import io
import os

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="True / False Image Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # okay for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Model setup
# -----------------------------
MODEL_PATH = "true_false_model.pth"

print("Current folder:", os.getcwd())
print("Looking for model:", os.path.abspath(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {os.path.abspath(MODEL_PATH)}")

print("Loading model...")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print("Model loaded successfully.")

# Must match your training folder order:
# false = 0, true = 1
classes = ["FALSE", "TRUE"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Image prediction API is running",
        "docs": "http://127.0.0.1:8000/docs",
        "endpoint": "POST /predict"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("Received file:", file.filename)

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {
            "error": "Invalid image file"
        }

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    result = classes[pred.item()]
    confidence_value = round(confidence.item(), 4)

    print("Prediction:", result, "Confidence:", confidence_value)

    return {
        "filename": file.filename,
        "result": result,
        "confidence": confidence_value
    }