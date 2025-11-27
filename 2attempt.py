import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from sklearn.metrics import f1_score, recall_score, precision_score

# Paths
image_folder = r"D:\Downloads\archive (2)\images\images_normalized"
csv_projections = r"D:\Downloads\archive (2)\indiana_projections.csv"
csv_reports = r"D:\Downloads\archive (2)\indiana_reports.csv"
model_save_path = r"D:\Downloads\archive (2)\resnet50_xray.pth"  # Save path for model

# Load CSV files
df_projections = pd.read_csv(csv_projections)
df_reports = pd.read_csv(csv_reports)

# Merge on UID
df = df_projections.merge(df_reports, on="uid")

# Selecting frontal images only (optional, modify as needed)
df = df[df['projection'] == 'Frontal']

# Extract filenames and labels
image_files = df['filename'].tolist()
labels = df.iloc[:, 1:].values  # Assuming multi-labels start from 1st column

# Custom Dataset
class XrayDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(image_folder, self.image_files[idx] + ".png")  # Modify extension if needed
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create Dataset
dataset = XrayDataset(image_files, labels, transform)

# Train-Validation Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model: ResNet50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)

# Modify classifier for multi-label classification
num_classes = labels.shape[1]
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes),
    nn.Sigmoid()  # Multi-label classification
)
model.to(device)

# Loss and Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the model (for evaluation or later use)
model.load_state_dict(torch.load(model_save_path))
model.to(device)
model.eval()
print("Model loaded for evaluation.")

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()  # Convert probabilities to binary values
        all_preds.append(preds.cpu().numpy())
        all_labels.append(targets.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Compute Metrics
f1 = f1_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")
precision = precision_score(all_labels, all_preds, average="macro")

print(f"F1 Score: {f1:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"Precision Score: {precision:.4f}")
