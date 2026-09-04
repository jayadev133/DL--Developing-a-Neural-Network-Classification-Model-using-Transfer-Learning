# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM

To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset

To develop an image classification model using a pretrained VGG19 neural network and transfer learning to classify chip images into two categories: **defect** and **notdefect**.

The dataset used for this experiment is **chip_data.zip**. It contains two classes:

- `defect`
- `notdefect`

The dataset is extracted and loaded using PyTorch's `ImageFolder` dataset. The images are resized to **224 × 224 pixels** and converted into tensors before being passed to the neural network.

## Neural Network Model

The pretrained **VGG19** architecture is used for transfer learning.

```text
                         Input Image
                              |
                              v
                       Resize 224 x 224
                              |
                              v
                    Pretrained VGG19
                              |
                              v
                  Feature Extraction
                              |
                              v
                           Flatten
                              |
                              v
                   Fully Connected Layer
                              |
                              v
                    Classification Layer
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                  defect            notdefect
```

## DESIGN STEPS

### STEP 1:

Import the required libraries such as PyTorch, Torchvision, NumPy, Matplotlib, Seaborn, and Scikit-learn.

### STEP 2:

Define the image preprocessing transformation by resizing all input images to **224 × 224 pixels** and converting them into PyTorch tensors.

### STEP 3:

Extract the `chip_data.zip` dataset and load the training and testing images using PyTorch's `ImageFolder` dataset.

### STEP 4:

Load the pretrained **VGG19** architecture and modify the final fully connected layer according to the number of classes in the dataset.

### STEP 5:

Freeze the pretrained VGG19 feature extraction layers and train the classification layer using the **CrossEntropyLoss** loss function and **Adam optimizer**.

### STEP 6:

Evaluate the trained model using the test dataset and generate the training loss plot, confusion matrix, classification report, and predictions for new sample images.

## PROGRAM

### Name:

Jayadev Pallinti

### Register Number:

212223240058

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import zipfile

# Define Image Transformation

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Extract Dataset

zip_path = "chip_data.zip"
extract_path = "./data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")

# Load Training and Testing Dataset

dataset_path = "./data/dataset/"

train_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/test",
    transform=transform
)

print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))

# Create DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Load Pretrained Model and Modify for Transfer Learning

model = models.vgg19(weights="DEFAULT")

# Modify the final fully connected layer to match the dataset classes

num_classes = len(train_dataset.classes)

model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features,
    num_classes
)

# Freeze VGG19 Feature Extraction Layers

for param in model.features.parameters():
    param.requires_grad = False

# Select Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("Using device:", device)

# Include the Loss Function and Optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# Train the Model

num_epochs = 5

train_losses = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {epoch_loss:.4f}"
    )

# Training Loss Plot

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, num_epochs + 1),
    train_losses,
    marker='o'
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Vs Epoch")

plt.grid(True)

plt.show()

# Evaluate the Model

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

# Calculate Test Accuracy

accuracy = (
    np.array(all_predictions) ==
    np.array(all_labels)
).mean()

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

# Generate Confusion Matrix

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=train_dataset.classes,
    yticklabels=train_dataset.classes
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.show()

# Generate Classification Report

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=train_dataset.classes
    )
)

# New Sample Data Prediction

model.eval()

sample_index = 55

image, actual_label = test_dataset[sample_index]

input_image = image.unsqueeze(0).to(device)

with torch.no_grad():

    output = model(input_image)

    _, predicted_label = torch.max(
        output,
        1
    )

predicted_label = predicted_label.item()

print("Sample Index:", sample_index)

print(
    "Actual Class:",
    test_dataset.classes[actual_label]
)

print(
    "Predicted Class:",
    test_dataset.classes[predicted_label]
)

# Display Sample Image

plt.figure(figsize=(5, 5))

plt.imshow(
    image.permute(1, 2, 0)
)

plt.title(
    f"Actual: {test_dataset.classes[actual_label]}\n"
    f"Predicted: {test_dataset.classes[predicted_label]}"
)

plt.axis("off")

plt.show()
```

### OUTPUT

## Confusion Matrix

<img width="655" height="485" alt="image" src="https://github.com/user-attachments/assets/0cb1c607-24d1-45f8-b6cd-47dc510abff9" />

## Classification Report
<img width="626" height="182" alt="image" src="https://github.com/user-attachments/assets/d7b2c841-e6af-4981-8030-daee778bff0f" />


### New Sample Data Prediction
<img width="350" height="351" alt="image" src="https://github.com/user-attachments/assets/0b481d57-7a58-4e37-8092-112e43983e2a" />


## RESULT

The VGG19 transfer learning model successfully classified chip images into defect and notdefect classes. The model achieved a 90.08% test accuracy, correctly classifying 109 out of 121 images. Thus, the model was successfully implemented for chip defect classification.
