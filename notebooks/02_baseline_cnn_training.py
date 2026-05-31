# Converted from 02_baseline_cnn_training.ipynb


# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import sys
sys.path.append("../src")

from model import SimpleCNN

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
data_dir = "../data/chest_xray"

batch_size = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# %%
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Classes:", train_dataset.classes)
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# %%
model = SimpleCNN().to(device)
print(model)

# %%
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return average_loss, accuracy

# %%
num_epochs = 5

train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"- loss: {train_loss:.4f} "
        f"- accuracy: {train_accuracy:.4f}"
    )

# %%
import os

os.makedirs("../outputs/models", exist_ok=True)

torch.save(model.state_dict(), "../outputs/models/baseline_cnn.pth")
