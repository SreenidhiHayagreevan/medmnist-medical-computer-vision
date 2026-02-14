# Computer Vision for Medical Image Analysis using MedMNIST

## Overview

This repository presents a series of Computer Vision experiments on standardized biomedical imaging datasets from MedMNIST. The project focuses on 2D and 3D medical image classification using deep convolutional neural networks, residual architectures, transfer learning, explainable AI techniques (Grad-CAM), and model optimization through quantization.

The work explores the complete Computer Vision pipeline for medical AI systems:
- Volumetric (3D) medical image classification
- 2D radiographic image analysis
- Transfer learning with pretrained vision models
- Residual network design
- Visual explainability for clinical trust
- Model compression for deployment

These experiments simulate real-world medical imaging workflows where accuracy, interpretability, and deployment constraints are critical.

---

# Problem 1: 3D Computer Vision for CT-Based Nodule Classification

## Objective
Develop a 3D Convolutional Neural Network (3D-CNN) for volumetric medical image classification using CT scan data.

## Dataset
- NoduleMNIST3D (lung CT nodules)
  OR
- AdrenalMNIST3D (adrenal gland CT scans)

These datasets contain 3D biomedical image volumes requiring volumetric convolution operations.

## Computer Vision Approach

- Designed a 3D CNN architecture from scratch
  - 3D Convolution layers (kernel size 3)
  - Batch Normalization for training stability
  - MaxPooling layers
  - Fully connected layers with Dropout (0.5)
- Trained for 100+ epochs with Early Stopping
- Evaluated using Accuracy and AUC
- Compared against MedMNIST published benchmarks

## CV Concepts Demonstrated

- Volumetric (3D) convolution
- Feature extraction from CT volumes
- Regularization in deep vision networks
- Performance benchmarking in medical imaging

---

# Problem 2: Advanced 2D Computer Vision for Radiographic Image Analysis

## Objective
Explore deep learning techniques for 2D medical image classification, incorporating augmentation, transfer learning, explainability, and deployment optimization.

## Dataset
- ChestMNIST (X-ray images)
  OR
- RetinaMNIST (retinal imaging)

These are 2D medical image classification tasks representative of radiology workflows.

---

## Part A – Custom CNN + Image Augmentation

### Computer Vision Implementation

- Built a deep CNN with:
  - Multiple convolutional layers
  - Batch normalization
  - Dropout regularization
- Applied image augmentation:
  - Rotation
  - Horizontal flipping
  - Scaling / transformations
- Evaluated model performance with and without augmentation

### CV Insight
Data augmentation improves generalization and robustness in medical Computer Vision systems where dataset size is limited.

---

## Part B – Transfer Learning with VGG16

### Computer Vision Implementation

- Loaded ImageNet-pretrained VGG16
- Replaced classifier head with:
  - Global Average Pooling
  - Fully connected layers
  - Final classification layer
- Compared:
  - Fully frozen backbone
  - Partial fine-tuning

### CV Insight
Transfer learning leverages pretrained visual feature extractors to improve performance and reduce training time in medical imaging tasks.

---

## Part C – Explainable Computer Vision (Grad-CAM)

- Generated Grad-CAM heatmaps
- Visualized discriminative regions influencing predictions
- Evaluated whether model attention aligns with meaningful anatomical structures

### Why This Matters
Explainability is essential in clinical Computer Vision systems to ensure transparency and build physician trust.

---

## Part D – Model Quantization for Vision Deployment

Applied:
- Post-training dynamic quantization
- Full integer quantization
- Quantization-aware training

Compared:
- Model size
- Accuracy trade-offs
- Deployment implications

### Deployment Insight
Quantization enables efficient deployment of Computer Vision models in constrained hardware environments.

---

# Problem 3: Residual Networks for Deep Computer Vision

## Objective
Develop a ResNet-style architecture from scratch for medical image classification.

## Computer Vision Implementation

- Implemented residual blocks:
  - Convolution → BatchNorm → ReLU
  - Skip connections
- Built multi-stage residual architecture
- Used Global Average Pooling and dense classifier
- Evaluated using Accuracy and AUC

## CV Concepts Demonstrated

- Residual learning
- Gradient flow improvement
- Deep feature hierarchy construction
- Stability in deep vision networks

---

# Technical Stack

- Python
- PyTorch
- TensorFlow / Keras
- MedMNIST
- Grad-CAM
- TensorFlow Model Optimization Toolkit

---

# Computer Vision Skills Demonstrated

- 2D and 3D medical image processing
- CNN and ResNet architecture design
- Transfer learning for vision tasks
- Explainable AI for medical imaging
- Model compression for deployment
- Performance benchmarking (AUC, Accuracy)
- Early stopping and training stability

---

# Conclusion

This project demonstrates end-to-end Computer Vision development for medical image analysis, from custom CNN de
