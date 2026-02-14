# Medical Image Classification & Model Optimization using MedMNIST Dataset

## Overview

This repository contains deep learning – computer vision experiments on standardized biomedical imaging datasets from **MedMNIST**. The project explores 2D and 3D medical image classification using custom CNN architectures, transfer learning, residual networks, explainability techniques (Grad-CAM), and model optimization via quantization.

The objective of this work is to evaluate different deep learning strategies for medical image analysis, benchmark performance using AUC and Accuracy, and analyze model behavior in safety-critical contexts.

The repository includes three main experiments:

- 3D CNN for volumetric CT classification
- 2D CNN with augmentation, transfer learning, Grad-CAM, and quantization
- ResNet architecture built from scratch for biomedical imaging

---

# 1: 3D CNN for NoduleMNIST3D

## Goal
Develop a 3D Convolutional Neural Network from scratch for volumetric medical image classification using MedMNIST3D datasets.

## Dataset
- **NoduleMNIST3D** (lung CT nodules)

This dataset consists of 3D biomedical CT image volumes used for binary malignancy classification.

## What Was Implemented

- Custom 3D CNN architecture
  - 3 convolutional blocks (Conv3D + BatchNorm + MaxPooling3D)
  - Fully connected classifier head
  - Dropout (0.5) for regularization
- Class imbalance handling using computed class weights
- Training for up to 100 epochs with Early Stopping
- Evaluation using Accuracy and AUC
- Learning curve visualization

## Final Test Results

| Model | Test Accuracy | Test AUC | Test Loss |
|--------|--------------|----------|----------|
| Custom 3D CNN | **0.8355** | **0.8720** | 0.4020 |

### Interpretation
- Achieved strong discriminative performance on volumetric CT data.
- AUC > 0.87 indicates reliable separation between benign and malignant nodules.
- Early stopping effectively prevented overfitting.

---

# 2: 2D CNN, Transfer Learning, Grad-CAM & Quantization (ChestMNIST)

## Goal
Explore advanced model development techniques for 2D biomedical image classification, including augmentation, transfer learning, explainability, and deployment optimization.

## Dataset
- **ChestMNIST** (multi-label chest X-ray dataset)

---

## Part A – Custom Deep CNN (Without Augmentation)

- 4 convolutional layers
- Batch normalization
- Dropout regularization
- BCEWithLogitsLoss
- Early stopping

### Final Test Results (No Augmentation)

| Metric | Value |
|--------|--------|
| Test Accuracy | **0.9479** |
| Test AUC | **0.7606** |

### Interpretation
- High classification accuracy on multi-label X-ray dataset.
- AUC reflects moderate separability across multiple pathologies.
- Demonstrates effective baseline 2D medical computer vision modeling.

---

## Part B – Transfer Learning (VGG16)

- Loaded ImageNet-pretrained VGG16
- Replaced classifier head with:
  - Global Average Pooling
  - Fully connected layers
  - Sigmoid output layer
- Compared frozen backbone vs partial fine-tuning

### Insight
Transfer learning accelerated convergence and improved feature extraction stability on limited medical imaging data.

---

## Part C – Grad-CAM (Explainability)

- Generated Grad-CAM heatmaps
- Visualized discriminative regions in X-ray images
- Verified model attention aligns with meaningful anatomical structures

### Why This Matters
Explainability improves trust and interpretability in medical AI systems.

---

## Part D – Model Quantization

Applied:
- Dynamic quantization
- Static quantization
- Quantization-aware training (QAT)

Compared:
- Model size reduction
- Accuracy trade-offs
- Deployment feasibility

### Insight
Quantization reduced model footprint while preserving predictive performance, supporting efficient deployment.

---

# 3: ResNet Architecture from Scratch (3D ResNet on NoduleMNIST3D)

## Goal
Develop a Residual Network (ResNet3D) architecture from scratch for biomedical volumetric image classification.

## Implementation

- Residual blocks:
  - Conv3D → BatchNorm → ReLU
  - Skip connections
- Multi-stage residual stacking
- Global Average Pooling
- Sigmoid output layer
- Early stopping and learning rate scheduling

## Final Test Results

| Model | Test Accuracy | Test AUC |
|--------|--------------|----------|
| Custom ResNet3D | **0.7697** | ~0.7035* |

\*Approximate AUC observed during evaluation phase in notebook logs.

### Interpretation
- Residual connections improved training stability.
- Demonstrated deeper 3D architecture design.
- Achieved competitive volumetric classification performance.

---

# Technical Stack

- Python
- PyTorch
- TensorFlow / Keras
- MedMNIST
- Matplotlib
- Grad-CAM
- TensorFlow Model Optimization Toolkit

---

# Key Skills Demonstrated

- 2D and 3D medical image processing
- CNN and ResNet architecture design
- Transfer learning
- Explainable AI (Grad-CAM)
- Model quantization & deployment awareness
- Performance benchmarking (AUC, Accuracy)
- Class imbalance handling
- Early stopping and training stability techniques

---

# Conclusion

This project explores the full lifecycle of medical image computer vision model development:

1. Architecture design (CNN, ResNet)
2. Training and evaluation
3. Generalization improvement via augmentation
4. Transfer learning strategies
5. Model interpretability using Grad-CAM
6. Deployment optimization via quantization

The experiments collectively demonstrate practical deep learning applications in biomedical computer vision, emphasizing performance, interpretability, and production readiness.
