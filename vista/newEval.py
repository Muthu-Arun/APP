import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import csv

# Configuration
IMAGE_FOLDER = "./train/fake/"  # Change this to your test folder
MODEL_PATH = "deepfake_detector.pth"
OUTPUT_CSV = "new_predictions.csv"

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 1),
    torch.nn.Sigmoid()
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode

# Process images and predict
results = []
for image_name in os.listdir(IMAGE_FOLDER):
    image_path = os.path.join(IMAGE_FOLDER, image_name)

    # Ensure it's an image
    if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(image).item()  # Get probability

    # Store results
    results.append([os.path.splitext(image_name)[0], output])

# Save results to CSV
with open(OUTPUT_CSV, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "probability"])
    writer.writerows(results)

print(f"Predictions saved to {OUTPUT_CSV}")
