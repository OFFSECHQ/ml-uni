# Project Summary

## Overview
This project implements a sports celebrity image classification system using machine learning. The system identifies athletes from uploaded images by detecting and analyzing facial features.

## Image Processing Pipeline

### 1. Face Detection
When analyzing an image, the system first identifies people using facial recognition. Images may contain multiple faces, and faces can sometimes be obstructed or unclear. The preprocessing pipeline follows these steps:

1. **Detect faces** in the uploaded image
2. **Detect eyes** within each detected face
3. **Validate the detection** - only keep images where at least 2 eyes are detected
4. **Discard invalid images** that don't meet the criteria

### 2. Haar Cascade Classifier
The system uses **Haar Cascade**, a machine learning-based approach for object detection in images. This method:
- Was proposed by Paul Viola and Michael Jones in 2001
- Is widely implemented in OpenCV
- Efficiently detects objects like faces, eyes, and other features
- Works by analyzing patterns of light and dark regions in images

### 3. Wavelet Transform Preprocessing
After face detection, the system applies **wavelet transform** as a key feature extraction technique:

**Why Wavelet Transform?**
- In wavelet-transformed images, **edges are clearly visible**
- These edges provide crucial clues about facial features:
  - Eyes
  - Nose
  - Lips
  - Face contours

**Feature Combination:**
The classifier uses both:
- **Raw pixel data** from the original image
- **Wavelet-transformed features** that highlight facial structure

This dual-feature approach improves classification accuracy by combining color/texture information with structural edge information.

### 4. Image Cropping and Storage
Once validation passes:
1. The face region is **cropped** from the original image
2. Images with detected faces (≥2 eyes) are **saved** for training
3. Cropped images in the `cropped` folder are used for model training

## Training Data Preparation

### Feature Vector Construction
The system prepares training data by combining raw and wavelet-transformed features:

**For each image:**

1. **Raw Image Processing:**
   - Read the image
   - Resize to **32×32 pixels**
   - Flatten to shape `(32×32×3,)` = `(3072,)` (RGB channels)

2. **Wavelet Transform Processing:**
   - Apply wavelet transform (w2d) to extract texture features
   - Resize to **32×32 pixels**
   - Flatten to shape `(32×32,)` = `(1024,)`

3. **Feature Combination:**
   - Vertically stack both feature vectors
   - Final feature vector shape: `(3072 + 1024,)` = `(5120,)`
   - Append to feature matrix **X**

4. **Label Assignment:**
   - Look up the person's name in `class_dict`
   - Assign the corresponding class label
   - Append to label vector **y**

### Final Training Data Structure
- **X (Features):** Matrix where each row is a 5120-dimensional feature vector
- **y (Labels):** Vector of class labels corresponding to each athlete

## Model Output
The trained classifier can identify the following athletes:
- Cristiano Ronaldo (Footballer)
- Gukesh Dommaraju (Chess Grandmaster)
- Neeraj Chopra (Javelin Thrower)
- PV Sindhu (Badminton Player)
- Shubman Gill (Cricket Player)

The model outputs confidence scores for each athlete, allowing the system to determine the most likely match for an uploaded image.

