#Comparative Analysis of CNN and ScatNet in Lung Cancer Detection

This project implements and compares two models — a standard Convolutional Neural Network (CNN) and a Scattering Network (ScatNet) — for the binary classification of lung cancer histopathological images. The dataset used consists of grayscale images labeled as **adenocarcinoma** or **benign**.

## 📁 Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images)
- **Preprocessing**:
  - Converted to grayscale
  - Resized to 64x64
  - Normalized to [-1, 1]

## 🧠 Models

### 1. CNN
- Architecture: 3 Convolutional Layers + ReLU + BatchNorm + MaxPooling + Fully Connected Layers
- Used Dropout for regularization

### 2. ScatNet
- Feature extraction using 2D Scattering Transform (J=2, L=8)
- Output: 81 scattering maps → mean pooled → [batch_size, 81]
- Uses the same fully connected classifier as CNN
