# Number Plate Recognition System

This repository contains a **Number Plate Recognition (NPR)** system that detects and recognizes license plates from images or video streams. The system is designed to handle various challenges such as different lighting conditions and angles. It uses a deep learning-based approach, employing the YOLOv9 model for number plate detection and EasyOCR for optical character recognition (OCR).

## Table of Contents

- [Overview](#overview)
- [System Pipeline](#system-pipeline)
- [Preprocessing](#preprocessing)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

This system performs two main tasks:
1. **Number Plate Detection:** The YOLOv9 model detects and extracts bounding boxes around license plates from input images or video streams.
2. **Number Plate Recognition:** Once the number plate is detected, EasyOCR is used to recognize the alphanumeric characters on the plate.

The solution aims to work under various lighting conditions and angles, making it robust for real-world applications.

## System Pipeline

The system pipeline is as follows:

1. **Input Image/Video Stream:** The system takes an image or video stream as input.
2. **Number Plate Detection:**
   - The YOLOv9 model is applied to detect bounding boxes around potential number plates.
   - The output bounding boxes are filtered based on a confidence threshold to identify the most likely regions containing number plates.
3. **Number Plate Cropping:** The regions corresponding to the bounding boxes are cropped from the input image.
4. **Text Recognition:** The cropped number plate images are processed using EasyOCR to extract the alphanumeric characters.
5. **Output Results:** The detected number plate text is displayed along with the bounding box on the original image.

## Preprocessing

Preprocessing plays a crucial role in improving the system's robustness to various challenges such as lighting variations, image angles, and distortions. Below are the key preprocessing steps applied:

### 1. **Resizing:**
   - **Description:** Input images are resized to a fixed resolution (224x320) before passing through the model.
   - **Effect:** This standardizes the input size, ensuring that the model can effectively process images with different dimensions, improving accuracy and consistency.

### 2. **Image Normalization:**
   - **Description:** The pixel values of the image are normalized (scaled to a range of 0 to 1) before feeding them into the model.
   - **Effect:** Normalization helps the model learn patterns effectively, reducing sensitivity to lighting variations and ensuring uniformity in feature extraction.

### 3. **Histogram Equalization:**
   - **Description:** Histogram equalization is applied to adjust the contrast of the image.
   - **Effect:** This enhances image details, making number plates more distinguishable, especially under poor lighting conditions, and helps the model to detect plates more reliably.

### 4. **Image Augmentation:**
   - **Description:** Various augmentation techniques such as rotation, flipping, and scaling are applied to the images during training.
   - **Effect:** These augmentations allow the model to learn robust features that help in handling different orientations and angles of number plates in real-world scenarios.

### 5. **Color Channel Adjustments:**
   - **Description:** Adjustments to brightness, contrast, and saturation are applied dynamically.
   - **Effect:** These modifications help make the model more robust to variations in lighting conditions, ensuring that the system works well under both overexposed and underexposed conditions.

### 6. **Noise Reduction:**
   - **Description:** Some noise reduction techniques, such as Gaussian blur, are used to smooth images.
   - **Effect:** This helps to remove any irrelevant noise that might interfere with detecting the number plate, improving detection in noisy environments (e.g., images with low quality or motion blur).

By applying these preprocessing steps, the system is made more resilient to challenges like variations in lighting, number plate angles, and image quality. These techniques ensure that the model performs robustly in real-world conditions, where images may not always be clear or uniform.

## Dataset

https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
The dataset was split into training and validation sets to ensure proper evaluation of the model during training.

## Evaluation Metrics

The performance of the number plate detection and recognition system was evaluated using the following metrics:

- **YOLO Detection Metrics:**
  - **Precision:** The proportion of true positive detections out of all positive detections.
  - **Recall:** The proportion of true positive detections out of all actual objects in the image.
  - **mAP50:** Mean Average Precision at IoU=50.
  - **mAP50-95:** Mean Average Precision averaged over IoU values from 0.5 to 0.95.

- **OCR Performance Metrics:**
  - **Accuracy:** The percentage of correctly recognized characters compared to the ground truth.
  - **Speed:** The time taken to process each image, which is critical for real-time applications.

For number plate detection, YOLOv9 was used with the following configuration:
- **Epochs:** 25
- **GPU:** NVIDIA T4 on Google Colab

For OCR, EasyOCR was used, known to provide approximately 95% accuracy in recognizing alphanumeric characters from images.

## Installation

To set up the project, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/number-plate-recognition.git
   cd number-plate-recognition
