import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet
    transforms.RandomHorizontalFlip(),  # Augmentation for better generalization
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset from train/real and train/fake
dataset = datasets.ImageFolder(root="./vista/train/", transform=transform)

# Split dataset into train (85%) and validation (15%)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size])

# Create DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64,
                        shuffle=False, num_workers=4)

# Load a pretrained ResNet18 model
model = models.resnet18(pretrained=True)

# Unfreeze ALL layers for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Modify the last layer for binary classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),  # Single output neuron
    nn.Sigmoid()  # Output probability
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      weight_decay=1e-4)  # Better for generalization
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training Loop
num_epochs = 35  # Increase training time
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(
            device).unsqueeze(1)  # Shape [batch, 1]

        optimizer.zero_grad()
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    # Validation step
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(
                device), labels.float().to(device).unsqueeze(1)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            # Convert probabilities to 0 or 1
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {
          avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

    # Adjust learning rate if needed
    scheduler.step(avg_val_loss)

print("Training complete.")

# Save model
torch.save(model.state_dict(), "deepfake_detector.pth")
