# Etch-Media-Coding-Test

# YOLOv8 Car Make and Model Classification

This repository contains the code for training, fine-tuning, and deploying a YOLOv8 model for car make and model classification. The project uses TensorRT for model optimization and FastAPI for serving the model. 

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.8 or higher
- Docker (for containerization)
- Git (for cloning the repository)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Set Up the Environment
Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Download and Prepare the Dataset
The dataset should be in a .tgz format. Extract it and prepare the dataset:

```bash
import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
base_dir = '/path/to/made_model'  # Path where your dataset is extracted
train_dir = os.path.join(base_dir, 'train')  # Path where training data will be saved
val_dir = os.path.join(base_dir, 'val')  # Path where validation data will be saved

# Create new directories for train and val
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# List all the car model folders
car_models = os.listdir(base_dir)

# Split ratio for training and validation
split_ratio = 0.8  # 80% training, 20% validation

for car_model in car_models:
    car_model_path = os.path.join(base_dir, car_model)

    if os.path.isdir(car_model_path):
        images = [img for img in os.listdir(car_model_path) if img.endswith('.jpg')]
        if len(images) == 0:
            print(f"No images found in {car_model_path}. Skipping...")
            continue

        train_images, val_images = train_test_split(images, train_size=split_ratio, random_state=42)
        train_model_dir = os.path.join(train_dir, car_model)
        val_model_dir = os.path.join(val_dir, car_model)
        os.makedirs(train_model_dir, exist_ok=True)
        os.makedirs(val_model_dir, exist_ok=True)

        for img in train_images:
            src = os.path.join(car_model_path, img)
            dest = os.path.join(train_model_dir, img)
            shutil.move(src, dest)

        for img in val_images:
            src = os.path.join(car_model_path, img)
            dest = os.path.join(val_model_dir, img)
            shutil.move(src, dest)

print("Dataset reorganized into train and validation sets.")
```

### 4. Train and Optimize the YOLOv8 Model
Run the training script to fine-tune the YOLOv8 model:

```bash
import torch
from ultralytics import YOLO

# Load the YOLOv8 model (for classification)
model = YOLO('yolov8n-cls.pt')

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# If using GPU, move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training
model.train(
    data='/path/to/made_model',
    epochs=5,
    imgsz=224,
    batch=32,
    lr0=1e-3,
    augment=True,
    save=True,
    save_period=1,
    project='/path/to/save',
    name='car_classifier',
    exist_ok=True
)

# Validation
metrics = model.val()
print("Validation metrics: ", metrics)
```

Optimize the trained model with TensorRT:

```bash
import torch
import tensorrt as trt
from torch2trt import torch2trt

# Conversion function
def convert_to_tensorrt(pytorch_model_path, engine_save_path, max_batch_size=16, int8_mode=False, calibration_dataset=None):
    model = YOLO(pytorch_model_path)
    model.eval()
    model.to(device)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if int8_mode:
        if calibration_dataset is None or len(calibration_dataset) == 0:
            print("Calibration dataset is required for INT8 mode.")
        else:
            calibrator = EntropyCalibrator(calibration_dataset)  # Ensure this is defined
            config.int8_calibrator = calibrator
            config.set_flag(trt.BuilderFlag.INT8)

    x = torch.randn((max_batch_size, 3, 224, 224), device=device)
    if device.type == 'cuda':
        trt_model = torch2trt(model, [x], fp16_mode=False, int8_mode=int8_mode, max_batch_size=max_batch_size)
        with open(engine_save_path, "wb") as f:
            f.write(trt_model.engine.serialize())
        print(f"TensorRT engine saved at {engine_save_path}")
    else:
        print("TensorRT conversion is not supported on CPU.")

# Example usage
convert_to_tensorrt('/path/to/best.pt', '/path/to/best.engine', max_batch_size=16, int8_mode=True, calibration_dataset=calibration_dataset)
```

### 5. Deploy the Model Using FastAPI
Create a main.py file for the FastAPI application:

```bash
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import io

app = FastAPI()

# Load the model
model = YOLO('/path/to/best.engine')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = model(img)
    return {"results": str(results)}
```

### 6. Create a Dockerfile
Create a Dockerfile:

```bash
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80
EXPOSE 80

# Define environment variable
ENV NAME World

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

### 7. Build and Run the Docker Image
Build the Docker image:

```bash
docker build -t car-classifier .
```

Run the Docker container:

```bash
docker run -p 80:80 car-classifier
```
