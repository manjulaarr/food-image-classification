# 🍔 Food Image Classification using CNN

## 📌 Project Overview
This project implements an end-to-end deep learning pipeline to classify food images into predefined categories using **Convolutional Neural Networks (CNNs)** built with **PyTorch**.  
It leverages the **Food-101 Tiny dataset** and demonstrates how CNNs can learn hierarchical features directly from raw pixels.

---

## 🛠️ Tools & Technologies
- Python  
- PyTorch  
- NumPy  
- OpenCV (for preprocessing)  

---

## 🎯 Problem Statement
- Manual food image labeling is **time-consuming and error-prone**.  
- Automated recognition enables **diet tracking, nutritional analysis, and smart meal planning**.  
- Real-world variability (lighting, angles, portion sizes, backgrounds) makes classification challenging.  

---

## 📂 Dataset
- **Food-101 Tiny dataset**  
- Images resized to **128×128 RGB**  
- Augmentation: random flip, rotation, brightness adjustment  
- Folder-based structure (PyTorch `ImageFolder`)  
- Split: **80% training / 20% validation**  

---

## ⚙️ Model Architecture
- **Input:** 3 × 128 × 128  
- **Conv Blocks:**  
  - Conv2d → BatchNorm → ReLU → MaxPool  
  - Increasing filters: 32 → 64 → 128 → 256  
- **Fully Connected Layers:**  
  - Flatten → Linear(16384 → 512) → Dropout(0.5) → Linear(512 → N classes)  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam (lr = 0.0005)  
- **Epochs:** 25  

---

## 🔄 Workflow
1. Dataset loading  
2. Preprocessing & augmentation  
3. Train/validation split  
4. CNN training  
5. Model evaluation  
6. Save model (`food_cnn.pt`)  
7. Predict new food images  

---

## 📊 Results
- Training loss decreased steadily across epochs.  
- Validation accuracy improved consistently.  
- Model successfully classified food categories (e.g., **Pizza, Sushi, Ice Cream**).  

### Confusion Matrix (Sample)
![Confusion Matrix](confusion_matrix.png)

---

## 🚀 Future Scope
- Train on **full Food-101 dataset** for better generalization.  
- Extend to **calorie estimation** and nutritional prediction.  
- Deploy as a **mobile/web app** using ONNX or TorchScript.  

---

## 🙌 Contributors
- Sumith S (ENG23CS0202)  
- Thrisha P (ENG23CS0483)  
- Manjula R (ENG23CS0693)  

---
💡 This project demonstrates practical application of **CNNs in computer vision** and is part of my portfolio showcasing AI/ML skills.
