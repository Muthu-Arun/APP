import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from PIL import Image
import os
import csv

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load the trained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('deepfake_classifier.pth'))
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Step 2: Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225]),  # ImageNet mean and std
])

# Step 3: Function to predict the label of an image


def predict_image(image_path):
    # image = Image.open(image_path)
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None  # Skip corrupt images
    # Add batch dimension and move to device
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # 0 -> Fake, 1 -> Real
    return predicted.item()

# Step 4: Process all images in the directory and write results to CSV


def predict_and_write_to_csv(input_directory, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image_id', 'Label'])  # Write CSV header

        for filename in os.listdir(input_directory):
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
                image_path = os.path.join(input_directory, filename)
                # Get the image ID without extension
                image_id = os.path.splitext(filename)[0]
                # Predict label (0 = fake, 1 = real)
                label = predict_image(image_path)
                writer.writerow([image_id, label])  # Write to CSV

    print(f"Predictions saved to {output_csv}")


# Step 5: Call the function with your image directory and desired output CSV file
input_directory = r'C:\Users\amuth\Vista\dataset\dataset'  # Path to the directory with images
output_csv = 'predictions.csv'  # Output CSV file to save the results

predict_and_write_to_csv(input_directory, output_csv)
