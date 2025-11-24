Biometric Fingerprint Recognition System

A Deep Learning-based fingerprint identification system built with PyTorch and OpenCV. This project utilizes a Convolutional Neural Network (CNN) trained on the SOCOFing (Sokoto Coventry Fingerprint) dataset to identify individuals from biometric data.

📌 Overview

This repository contains a complete pipeline for training a fingerprint classifier and running inference on new samples. It handles image preprocessing (CLAHE, Gaussian Blur), data augmentation, and CNN-based classification.

Key Features:

Preprocessing Pipeline: Enhances ridge details using CLAHE (Contrast Limited Adaptive Histogram Equalization).

Custom CNN Architecture: Lightweight 3-layer Convolutional Network optimized for 128x128 grayscale inputs.

Robustness: Trained with random rotation augmentation to handle misaligned scans.

Inference Engine: Includes a standalone script to load trained weights and predict identities from new BMP images.

📂 Dataset

The project is designed for the SOCOFing dataset. You can download it from Kaggle.

Expected Directory Structure:

/path/to/SOCOFing/
├── Real/           # Contains unaltered .BMP fingerprint images
└── Altered/        # Contains Easy, Medium, Hard altered versions


🛠️ Installation & Requirements

Clone the repository (or ensure your script files are in one folder).

Install dependencies:

pip install torch torchvision opencv-python scikit-learn matplotlib numpy


Note: If you have a CUDA-capable GPU, ensure you install the appropriate version of PyTorch for faster training.

⚙️ Configuration

Open the training script and update the DATA_DIR variable to point to your dataset location:

# In your training script
DATA_DIR = r"path/to/your/SOCOFing"


You can also adjust the MAX_IDS variable to control how many unique identities (classes) to train on. The default is set to 50 for rapid prototyping.

🧠 Model Architecture

The model (SimpleFingerprintCNN) consists of the following layers:

Input: Grayscale Image (128x128x1)

Conv Block 1: 32 filters (3x3), BatchNorm, ReLU, MaxPool

Conv Block 2: 64 filters (3x3), BatchNorm, ReLU, MaxPool

Conv Block 3: 128 filters (3x3), BatchNorm, ReLU, MaxPool

Flatten

Fully Connected: 256 neurons + Dropout (0.3)

Output: Softmax over N classes (Person IDs)

🚀 Usage

1. Training the Model

Run the training script (e.g., in Jupyter Notebook or Python file). This will:

Load "Real" images.

Split data into Training (80%) and Validation (20%) sets.

Train for 8 Epochs.

Save the model artifacts:

fingerprint_cnn.pth (Model Weights)

label_encoder.pkl (ID-to-Label mapping)

2. Running Inference

To identify a person from a new fingerprint image, use the inference code block.

from inference import predict_fingerprint

# Path to a test image (Real or Altered)
image_path = "SOCOFing/Altered/Altered-Hard/10__M_Right_index_finger_CR.BMP"

# Run prediction
pred_id, confidence = predict_fingerprint(image_path)

print(f"Identified Person ID: {pred_id} with {confidence:.2f} confidence")


🖼️ Preprocessing Details

Before entering the CNN, every image undergoes:

Grayscale Conversion: Reduces dimensionality.

Resize: Standardized to 128x128 pixels.

Gaussian Blur (5x5): Reduces high-frequency sensor noise.

CLAHE: Enhances local contrast to make ridges distinct.

Normalization: Scales pixel values to range [-1, 1].

📝 License

This project is open-source. The SOCOFing dataset belongs to its respective owners.# fingerprint-detection
