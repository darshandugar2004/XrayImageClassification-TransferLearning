import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data directories
data_dir = r"D:\Downloads\xray_bone_fracture"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# Define transformations with advanced augmentations
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform["train"])
val_dataset = datasets.ImageFolder(val_dir, transform=transform["val"])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Compute class weights for handling imbalance
class_counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in train_dataset.classes]
class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=train_dataset.targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Load DenseNet121 model (better for medical images)
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),  # Dropout for regularization
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, 2)  # Output layer for binary classification
)
model = model.to(device)

# Define loss function with class weights and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR after 10 epochs


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        val_acc, precision, recall, f1 = evaluate_model(model, val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}% - Precision: {precision:.4f} - Recall: {recall:.4f} - F1-Score: {f1:.4f}")

    return model


# Evaluation function with precision, recall, F1-score
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return accuracy, precision, recall, f1


# Test-Time Augmentation (TTA) for improved predictions
def tta_predict(model, image, transforms_list):
    image = image.to(device)
    preds = []
    
    for transform in transforms_list:
        img_t = transform(image)
        img_t = img_t.unsqueeze(0)  # Add batch dimension
        output = model(img_t)
        preds.append(torch.softmax(output, dim=1).cpu().numpy())
    
    return np.mean(preds, axis=0)  # Average predictions


# Train and save the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)
torch.save(trained_model.state_dict(), "xray_fracture_densenet121.pth")
print("Model training complete and saved!")
