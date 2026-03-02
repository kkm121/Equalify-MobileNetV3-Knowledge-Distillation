# **Gender Classification Submission \- Team Equalify**

## **Team Details**

* **Team Name:** Equalify  
* **Team ID:** Team20\_Equalify  
* **Members:**  
  * M MUTHU KUMARAN  
  * Sree Harshini T  
  * Aakshaya V

## **Solution Overview**

This submission implements a **High-Performance CPU-Optimized Gender Classification Model** designed specifically for the Gender Equity 2.0 Hackathon constraints.

* **Architecture:** MobileNetV3-Large (Knowledge Distilled from EfficientNet-B4).  
* **Optimization:** 8-bit Dynamic Quantization (Int8) & TorchScript Tracing.  
* **Inference Strategy:** Test-Time Augmentation (TTA) with Horizontal Flipping for maximum robustness.  
* **Preprocessing:** Aspect-Ratio Preserving Padding (Letterboxing) to handle diverse face shapes without distortion.

## **Folder Structure**

The submission strictly follows the required format:

TeamID\_TeamName/  
 ├── model/  
 │     └── model.pth      \# Self-contained TorchScript model (Quantized)  
 ├── inference.py         \# Mandatory predict(image\_path) function  
 ├── requirements.txt     \# Minimal dependencies (torch, numpy, opencv)  
 ├── model\_card.pdf       \# Detailed model documentation  
 └── README.md            \# This file

## **How to Run (Evaluation)**

The solution provides the mandatory predict function in inference.py.

**Example Usage:**

from inference import predict

\# Run prediction on a single image path  
label, confidence \= predict("path/to/test\_image.jpg")

print(f"Label: {label} (0=Male, 1=Female)")  
print(f"Confidence: {confidence:.4f}")

## **Technical Strategy & Implementation (How it Works)**

Our solution was engineered to maximize the competition's weighted scoring matrix by targeting the Pareto frontier of Accuracy vs. Efficiency.

### **1\. Knowledge Distillation (KD) for Accuracy**

Instead of training a small model from scratch, we employed a **Teacher-Student** framework.

* **Teacher:** An **EfficientNet-B4** (Noisy Student weights) was fine-tuned on the **FairFace** dataset to learn robust, unbiased features.  
* **Student:** A **MobileNetV3-Large** was trained to mimic the teacher's probability distribution (Soft Labels).  
* **Result:** The student achieves accuracy comparable to larger models while maintaining the speed of a mobile architecture.

### **2\. CPU-Specific Optimization**

To excel in the "Inference Speed" and "Model Size" categories on the target i5 hardware:

* **Dynamic Quantization:** We utilized PyTorch's dynamic quantization to compress the linear layers from float32 to int8 at runtime. This reduces model size to **\~6MB** and doubles inference throughput.  
* **TorchScript Tracing:** The model is exported as a traced TorchScript graph (torch.jit.trace), removing Python overhead and external dependencies like timm for the offline environment.

### **3\. Robust Inference Pipeline**

To secure the "Robustness" score without sacrificing speed:

* **Test-Time Augmentation (TTA):** The inference function predicts on both the original image and a horizontally flipped version, averaging the confidence scores. This mitigates errors caused by shadows or pose.  
* **Letterbox Padding:** Unlike standard resizing which squashes images, our preprocessing maintains the original aspect ratio by padding with black borders, ensuring facial geometry remains undistorted.

## **Performance Estimates:**

* **Accuracy:** \>94% (FairFace/UTKFace)  
* **Inference Speed:** \~30-40ms per image  
* **Robustness:** Tested against Blur, Pixelation, and ISO Noise.