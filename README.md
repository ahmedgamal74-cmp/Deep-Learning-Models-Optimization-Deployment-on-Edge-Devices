# Optimize and Deploy Deep Learning Models on Edge Devices

Welcome to our official repository for the graduation project titled **"Optimize and Deploy Deep Learning Models on Edge Devices"**, developed by:
  
**Mentored by:** Si-Vision  

---

## 📘 Abstract

This project explores efficient deployment of deep learning models on resource-constrained edge devices, particularly the NVIDIA Jetson Nano. It addresses computational, memory, and energy limitations using a full pipeline of optimization techniques, including pruning, quantization, knowledge distillation, low-rank approximation, and TensorRT acceleration. We evaluated this pipeline on architectures like VGG11, MobileNetV2, EfficientNet, NASNet, and AlexNet, achieving up to 18× speedup with minimal accuracy loss.

---

## 📂 Table of Contents

- [Project Overview](#project-overview)
- [Optimized Models](#optimized-models)
- [Optimization Techniques](#optimization-techniques)
- [Deployment Tools](#deployment-tools)
- [Results](#results)
- [Environment](#environment)
- [Contributors](#contributors)

---

## 🚀 Project Overview

The project’s objective is to deploy deep learning models efficiently on the Jetson Nano using a scalable optimization pipeline. The goal is to retain near-original accuracy while achieving substantial reductions in model size and inference time.

---

## 🧠 Optimized Models

- **AlexNet**
- **VGG11 & VGG16**
- **MobileNetV2**
- **EfficientNet-B0**
- **NASNet-Mobile**
- **YOLOv8s**
- **LLMs** (e.g., Qwen2.5-0.5B, TinyLlama-1.1B)

---

## ⚙️ Optimization Techniques

### 🔧 Pruning
- Structured channel pruning
- Iterative pruning
- FastNAS pruning
- Torch-Pruning with dependency graphs

### 📏 Quantization
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- TensorRT precision calibration (INT8, FP16)

### 🧊 Low-Rank Approximation
- SVD-based layer compression
- CP-like decomposition for Conv2D

### 🧪 Knowledge Distillation
- Student-teacher training with NASNet
- Used for both classification and detection tasks

---

## 🧰 Deployment Tools

- **TensorRT** for high-speed inference
- **ONNX Runtime** as baseline
- **PyTorch** & **TensorFlow Lite** as training and conversion tools

---

## 📈 Results

- **MobileNetV2:** 9.88MB → 0.47MB, 31.9 FPS → 66.1 FPS, 92.6% accuracy retained  
- **VGG11:** 7× speedup with only 2% accuracy drop  
- **YOLOv8s:** Significant mAP improvements via distillation and quantization  
- **LLMs:** Benchmarked TinyLlama, DeepSeek, and others on Jetson Nano

📊 **Full Excel Results Sheet:** [View on OneDrive](https://docs.google.com/spreadsheets/d/1Db0EXINeAfmpou3PeocLsoQTGAfQ5cSlHY8U9TEEHKI/edit?gid=0#gid=0)

(See detailed tables in `/results` and thesis for full metrics)

---

## 🧪 Environment

- **Device:** NVIDIA Jetson Nano
- **Frameworks:** PyTorch, TensorFlow Lite, ONNX, TensorRT
- **Datasets:** CIFAR-10, ImageNet, COCO, Pascal VOC


---
