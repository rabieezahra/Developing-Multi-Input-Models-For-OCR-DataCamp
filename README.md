# OCRModel: Multimodal OCR for Insurance ID Classification

A PyTorch implementation of a multimodal Optical Character Recognition (OCR) model for digitizing historical insurance claim documents. The model classifies scanned ID images as **primary** or **secondary** IDs by combining visual information from document images with the associated **insurance type**.

## Overview

DigiNsure Inc. is digitizing historical insurance claim documents. Part of this effort involves improving the labeling of scanned IDs and identifying them as primary or secondary IDs.

This project uses **multi-modal learning** to improve classification. The model receives:

- **Image input**: 64×64 grayscale scanned document images
- **Type input**: insurance type — `home`, `life`, `auto`, `health`, or `other`

The output is a label indicating whether the ID is **primary** or **secondary**.

## Features

- Multimodal architecture combining image and categorical insurance-type features
- CNN-based image branch saved as `image_layer`
- Dedicated insurance-type branch
- End-to-end training in PyTorch
- Uses `CrossEntropyLoss` and `Adam`
- Trains for 10 epochs
- Works with GPU if available

## Model Architecture

The `OCRModel` consists of three main components:

### 1. Image branch — `image_layer`

A sequential CNN for 64×64 grayscale images:

```python
self.image_layer = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten()
)
```

### 2. Insurance-type branch

```python
self.type_layer = nn.Sequential(
    nn.Linear(num_types, 16),
    nn.ReLU()
)
```

### 3. Combined classifier

```python
self.classifier = nn.Sequential(
    nn.Linear(64 * 8 * 8 + 16, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_labels)
)
```

The image features and insurance-type features are concatenated before the final classifier.

## Dataset

The project expects a pickled dataset named:

```text
ocr_insurance_dataset.pkl
```

The dataset is loaded using `project_utils.ProjectDataset`.

Each sample contains:

- `img[0]`: image tensor, shape `(1, 64, 64)`
- `img[1]`: one-hot insurance type vector
- `lbl`: label for primary/secondary ID

Example:

```python
import pickle
from project_utils import ProjectDataset

dataset = pickle.load(open("ocr_insurance_dataset.pkl", "rb"))
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

Make sure `project_utils.py` and `ocr_insurance_dataset.pkl` are available in the project directory.

## Usage

### 1. Load the dataset

```python
import pickle
from project_utils import ProjectDataset

dataset = pickle.load(open("ocr_insurance_dataset.pkl", "rb"))
```

### 2. Create the model

```python
import torch
import torch.nn as nn

model = OCRModel(
    num_types=len(dataset.type_mapping),
    num_labels=len(dataset.label_mapping)
)
```

### 3. Define loss and optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 4. Train the model

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for (images, types), labels in train_loader:
        images = images.to(device)
        types = types.to(device)
        labels = labels.to(device)

        if labels.dim() > 1:
            labels = labels.argmax(dim=1)

        labels = labels.long().view(-1)

        optimizer.zero_grad()
        outputs = model(images, types)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
```

## Training Configuration

| Setting | Value |
|---|---|
| Loss function | `CrossEntropyLoss` |
| Optimizer | `Adam` |
| Learning rate | `1e-3` |
| Batch size | `64` |
| Epochs | `10` |
| Input image size | `64 × 64` |
| Image channels | `1` grayscale |

