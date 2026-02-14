# Medical Image Classification & Model Optimization using MedMNIST Dataset

## Overview

This repository contains deep learning - computer vission experiments on standardized biomedical imaging datasets from **MedMNIST**. The project explores 2D and 3D medical image classification using custom CNN architectures, transfer learning, residual networks, explainability techniques (Grad-CAM), and model optimization via quantization.

The objective of this work is to evaluate different deep learning strategies for medical image analysis, benchmark performance using AUC and Accuracy, and analyze model behavior in safety-critical contexts.

The repository includes three main experiments:

- 3D CNN for volumetric CT classification
- 2D CNN with augmentation, transfer learning, Grad-CAM, and quantization
- ResNet architecture built from scratch for biomedical imaging

---

# 1: 3D CNN for NoduleMNIST3D / AdrenalMNIST3D

## Goal
Develop a 3D Convolutional Neural Network from scratch for volumetric medical image classification using MedMNIST3D datasets.

## Dataset
- NoduleMNIST3D (lung CT nodules)  
  OR  
- AdrenalMNIST3D (adrenal gland CT scans)

These datasets consist of 3D biomedical image volumes used for classification tasks.

## What Was Implemented

- Custom 3D CNN architecture
  - Minimum 3 convolutional layers
  - Batch Normalization after each convolution
  - MaxPooling layers
  - Fully connected layers with Dropout (0.5)
- Training for 100+ epochs with Early Stopping
- Evaluation using Accuracy and AUC
- Learning curve visualization
- Performance comparison with MedMNIST benchmark

## Key Learning

This experiment demonstrates:
- Handling 3D volumetric medical data
- Designing CNN architectures from scratch
- Evaluating medical models using clinically relevant metrics
- Understanding overfitting and generalization in small biomedical datasets

---

# 2: 2D CNN, Transfer Learning, Grad-CAM & Quantization (ChestMNIST / RetinaMNIST)

## Goal
Explore advanced model development techniques for 2D biomedical image classification, including augmentation, transfer learning, explainability, and deployment optimization.

## Dataset
- ChestMNIST (X-ray classification)
  OR  
- RetinaMNIST (retinal imaging)

## Part A – Custom Deep CNN + Augmentation

- Designed a CNN with:
  - At least 3 convolutional layers
  - Batch normalization
  - Dropout regularization
- Applied image augmentation techniques:
  - Rotation
  - Flipping
  - Scaling / other transformations
- Compared model performance with and without augmentation
- Evaluated using:
  - AUC
  - Accuracy

### Insight
Augmentation improves robustness and generalization in medical image classification.

---

## Part B – Transfer Learning (VGG16)

- Loaded VGG16 pretrained on ImageNet
- Replaced classifier head with:
  - Global Average Pooling
  - Fully connected layers
  - Final classification layer
- Compared two strategies:
  - Freeze all convolutional layers
  - Freeze partial layers and fine-tune remaining layers
- Evaluated training stability, convergence speed, and performance

### Insight
Transfer learning significantly improves performance and training efficiency on small medical datasets.

---

## Part C – Grad-CAM (Explainability)

- Generated Grad-CAM heatmaps for selected samples
- Visualized model attention regions
- Analyzed whether highlighted regions align with meaningful anatomical structures

### Why This Matters
Explainability is critical in medical AI to ensure model decisions are interpretable and trustworthy.

---

## Part D – Model Quantization

Applied three quantization techniques:
- Post-training dynamic range quantization
- Full integer quantization
- Quantization-aware training (QAT)

Compared:
- Model size reduction
- Test accuracy impact
- Deployment trade-offs

### Insight
Quantization enables model compression with minimal accuracy degradation, supporting deployment in resource-constrained environments.

---

# 3: ResNet Architecture from Scratch (MedMNIST)

## Goal
Develop a Residual Network (ResNet) architecture from scratch for biomedical image classification.

## Implementation

- Built residual blocks with:
  - Convolution → BatchNorm → ReLU
  - Skip connections
- Constructed full ResNet-style architecture with:
  - Stacked residual sections
  - Global Average Pooling
  - Dense output layer
- Trained for minimum required epochs
- Evaluated performance using Accuracy and AUC

## Key Concepts Demonstrated

- Residual learning
- Deep network stability
- Gradient flow improvement via skip connections
- Comparison of deeper architectures vs standard CNNs

---

# Technical Stack

- Python
- PyTorch
- TensorFlow / Keras
- MedMNIST
- Matplotlib / Seaborn
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
- Early stopping and training stability techniques

---

# Conclusion

This project explores the full lifecycle of medical image model development:

1. Architecture design (CNN, ResNet)
2. Training and evaluation
3. Generalization improvement via augmentation
4. Transfer learning strategies
5. Model interpretability using Grad-CAM
6. Deployment optimization via quantization

The experiments collectively demonstrate practical deep learning applications in biomedical image analysis, emphasizing performance, interpretability, and production readiness.

---
