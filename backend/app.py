from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms
from PIL import Image
import torch
import io
import os

# ==========================================
# 🚀 FASTAPI APP INITIALIZATION
# ==========================================
app = FastAPI(title="Healthcare Disease Detection API")

# Allow frontend access (React, Node, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🖼️ FRONTEND BUILD (React Deployment Support)
# ==========================================
if os.path.exists("build"):
    app.mount("/static", StaticFiles(directory="build/static"), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse("build/index.html")

# ==========================================
# 🔧 MODEL CONFIGURATION
# ==========================================
MODEL_DIR = "model/models"

MODEL_PATHS = {
    "lung": os.path.join(MODEL_DIR, "lung_model.pth"),
    "liver": os.path.join(MODEL_DIR, "liver_model.pth"),
    "breast": os.path.join(MODEL_DIR, "breast_model.pth"),
}

CLASS_NAMES = {
    "lung": ["Normal", "Pneumonia"],
    "liver": ["Normal", "Tumor"],
    "breast": ["Benign", "Malignant"],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 🧠 MODEL LOADER
# ==========================================
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=DEVICE)

    # handle checkpoints wrapped in dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        checkpoint = {
            k.replace("module.", ""): v
            for k, v in checkpoint.items()
        }

    model.load_state_dict(checkpoint, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded model: {model_path}")
    return model


# ==========================================
# 📦 LOAD ALL MODELS
# ==========================================
models_dict = {}

for disease, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            models_dict[disease] = load_model(path, len(CLASS_NAMES[disease]))
        except Exception as e:
            print(f"❌ Failed to load model {disease}: {e}")
    else:
        print(f"⚠️ Missing model file for: {disease}")


# ==========================================
# 🔄 IMAGE TRANSFORM
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
# 🔍 PREDICTION FUNCTION
# ==========================================
def predict_image(model, image_bytes, disease_type):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Invalid image file. Upload PNG/JPG only.")

    img_t = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        class_name = CLASS_NAMES[disease_type][predicted.item()]

    return class_name


# ==========================================
# 🌐 API ROUTES
# ==========================================
@app.get("/api")
def root():
    return {"message": "Healthcare Detection API Running!"}


@app.post("/predict/{disease_type}")
async def predict_disease(disease_type: str, file: UploadFile = File(...)):
    disease_type = disease_type.lower().strip()

    if disease_type not in models_dict:
        return JSONResponse(
            content={"error": f"Invalid disease '{disease_type}'. Choose lung, liver, breast."},
            status_code=400
        )

    contents = await file.read()

    try:
        prediction = predict_image(models_dict[disease_type], contents, disease_type)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    response = {
        "disease": disease_type.capitalize(),
        "prediction": prediction
    }

    print(f"🔎 Prediction result → {response}")
    return JSONResponse(content=response)
