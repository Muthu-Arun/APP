import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from sklearn.metrics import accuracy_score
import os
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformations for preprocessing images (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225]),  # Mean and std for ImageNet
])

# Load dataset using ImageFolder
train_data = datasets.ImageFolder(root='./vista/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Check dataset structure
print(f"Classes: {train_data.classes}")
print(f"Number of images: {len(train_data)}")
# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)

# Move model to the selected device
model = model.to(device)
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(f"Epoch {
          epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

torch.save(model.state_dict(), 'deepfake_classifier.pth')
