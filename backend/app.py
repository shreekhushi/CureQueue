import os
import io
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image

# Initialize App
app = FastAPI(title="Healthcare Disease Detection API")

# ==========================================
# 🌐 CORS CONFIGURATION
# ==========================================
# This allows your Vercel frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
# Verify this matches your folder structure exactly
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
# 🧠 MEMORY OPTIMIZATION (Global Variables)
# ==========================================
# We store only ONE model in memory at a time to prevent crashes on free servers
current_model = None
current_model_name = ""

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
# 🛠️ UTILITY FUNCTIONS
# ==========================================
def load_model_architecture(num_classes):
    """Creates the ResNet18 architecture"""
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def get_model(disease_type):
    """
    Loads the requested model into memory only when needed.
    Unloads other models to save RAM.
    """
    global current_model, current_model_name

    # If the requested model is already loaded, return it immediately
    if current_model_name == disease_type and current_model is not None:
        return current_model

    print(f"🔄 Loading model for: {disease_type}...")
    
    path = MODEL_PATHS.get(disease_type)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    num_classes = len(CLASS_NAMES[disease_type])
    
    # Initialize architecture
    model = load_model_architecture(num_classes)
    
    # Load weights
    checkpoint = torch.load(path, map_location=DEVICE)
    
    # Handle dictionary checkpoints if necessary
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        # Clean up module prefix if present
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    
    model.load_state_dict(checkpoint, strict=False)
    model.to(DEVICE)
    model.eval()

    # Update global state
    current_model = model
    current_model_name = disease_type
    
    print(f"✅ Successfully loaded {disease_type} model")
    return current_model

def predict_image(model, image_bytes, disease_type):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Invalid image file. Please upload a valid image.")

    img_t = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        class_name = CLASS_NAMES[disease_type][predicted.item()]

    return class_name

# ==========================================
# 🚀 API ROUTES
# ==========================================
@app.get("/")
def home():
    return {"message": "Healthcare Disease Detection API is Live! Use the /predict endpoints."}

@app.post("/predict/{disease_type}")
async def predict_disease(disease_type: str, file: UploadFile = File(...)):
    disease_type = disease_type.lower().strip()

    if disease_type not in MODEL_PATHS:
        return JSONResponse(
            content={"error": f"Invalid disease type '{disease_type}'. Valid options: {list(MODEL_PATHS.keys())}"},
            status_code=400
        )

    try:
        # 1. Load the specific model (and clear others from memory)
        model = get_model(disease_type)
        
        # 2. Read file
        contents = await file.read()
        
        # 3. Predict
        prediction = predict_image(model, contents, disease_type)
        
        response = {
            "disease": disease_type.capitalize(),
            "prediction": prediction
        }
        print(f"🔎 Result: {response}")
        return JSONResponse(content=response)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ==========================================
# 🏁 EXECUTION ENTRY POINT (REQUIRED FOR RENDER)
# ==========================================
if __name__ == "__main__":
    # Render sets the PORT environment variable. default to 10000 if not found.
    port = int(os.environ.get("PORT", 10000))
    # Run the app binding to 0.0.0.0 (required for external access)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)