# Converted from 03_model_evaluation.ipynb


# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

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

test_dataset = datasets.ImageFolder(
    root=f"{data_dir}/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Classes:", test_dataset.classes)
print("Test images:", len(test_dataset))

# %%
model = SimpleCNN().to(device)

model.load_state_dict(
    torch.load("../outputs/models/baseline_cnn.pth", map_location=device)
)

model.eval()

print("Model loaded successfully")

# %%
def evaluate_model(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return average_loss, accuracy, all_predictions, all_labels

# %%
criterion = nn.CrossEntropyLoss()

test_loss, test_accuracy, y_pred, y_true = evaluate_model(
    model,
    test_loader,
    criterion,
    device
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# %%
import os

os.makedirs("../outputs/figures", exist_ok=True)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix - Baseline CNN")

plt.tight_layout()

fig.savefig("../outputs/figures/confusion_matrix.png", dpi=300, bbox_inches="tight")

plt.show()

# %%
print(classification_report(
    y_true,
    y_pred,
    target_names=test_dataset.classes
))

# %% [markdown]
# The baseline CNN achieved 71.96% test accuracy, but the confusion matrix shows a strong bias toward predicting PNEUMONIA. Pneumonia recall was high (0.99), while NORMAL recall was low (0.26), indicating that accuracy alone is insufficient for evaluating this model. This motivates the Grad-CAM analysis in the next notebook, where inspected pneumonia predictions are based on meaningful lung regions or possible dataset artifacts.
