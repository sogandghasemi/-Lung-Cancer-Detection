# Comparative Analysis of CNN and ScatNet for Lung Cancer Detection with Explainable AI

This project presents a comparative study between **Convolutional Neural Networks (CNNs)** and **Scattering Networks (ScatNet)** for binary classification of **histopathological lung cancer images**. The task focuses on distinguishing **adenocarcinoma** from **benign** tissue samples.

In addition to classification performance, the project emphasizes **model interpretability** by integrating **Explainable AI (XAI)** techniques, enabling analysis of how each model makes its predictions.

---

## 📌 Project Objectives

- Compare **learned features (CNN)** with **handcrafted wavelet-based features (ScatNet)**
- Evaluate model performance using **5-fold cross-validation**
- Analyze **interpretability and transparency** using XAI methods
- Study the trade-off between **accuracy and explainability** in medical imaging models

---

## 📁 Dataset

- **Source**:  
  Kaggle – Lung Cancer Histopathological Images  
  https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images

- **Classes**:
  - Adenocarcinoma (label = 1)
  - Benign (label = 0)

- **Evaluation Setup**:
  - Held-out test set: 2000 images  
  - Remaining data used for 5-fold cross-validation

---

## 🧪 Preprocessing

### CNN Pipeline
- Images loaded in **RGB format**
- Resized to **224 × 224**
- Data augmentation with **random horizontal flips**
- Normalization to range **[-1, 1]**
- Fixed random seeds for reproducibility

### ScatNet Pipeline
- Images converted to **grayscale**
- Resized to **64 × 64**
- Feature extraction using **2D Scattering Transform**
  - Scales: `J = 2`
  - Orientations: `L = 8`
- Extracted **81 scattering coefficients** per image
- Mean pooling applied to obtain fixed-length feature vectors

---

## 🧠 Models

### 1️⃣ Convolutional Neural Network (CNN)

- Architecture:
  - Three convolutional layers
  - ReLU activations
  - Max pooling
  - Dropout for regularization
  - Fully connected classification head
- Trained using **5-fold cross-validation**
- Early stopping based on validation loss

---

### 2️⃣ Scattering Network (ScatNet)

- Uses **fixed wavelet filters** for feature extraction
- Scattering features passed to the **same fully connected classifier** used by the CNN
- Enables a fair comparison between learned and handcrafted features
- Trained using **5-fold cross-validation**

---

## 🔍 Explainable AI (XAI)

### CNN Explainability
- **SHAP (Shapley Additive Explanations)** using blur-based masking
- **Saliency Maps** based on gradient information

### ScatNet Explainability
- Applied **Shapley value sampling** to scattering coefficients
- Identified low-frequency averaging components as the most influential features
- Analysis suggests potential feature reduction without loss of performance

---

## 📊 Results

### CNN Performance (Held-Out Test Set)
- **Accuracy**: 100%
- **F1-score**: 1.00

### ScatNet Performance (Held-Out Test Set)
- **Accuracy**: 89.30%
- **F1-score (Macro)**: 0.89

### Key Observations
- CNN achieved higher accuracy but showed mild overfitting in some folds
- ScatNet exhibited smoother learning curves and better generalization
- ScatNet provides improved interpretability due to fixed, mathematically defined filters

---

## 🧠 Discussion

- CNNs offer strong performance but limited interpretability
- ScatNet provides a more transparent and explainable framework
- The comparison highlights a practical trade-off between **performance and transparency** in medical AI

---

## 🚀 Future Work

- Feature selection using SHAP scores
- Combining mean and standard deviation pooling
- Exploring deeper or attention-based classifiers
- Developing hybrid CNN–ScatNet models

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Kymatio
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📚 References

- Dataset: https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images  
- SHAP Documentation: https://shap.readthedocs.io/en/latest/index.html
