# Teeth Diseases Classification with ResNet50

This repository explains how to implement a Convolutional Neural Network (CNN) to classify 7 types of teeth diseases. The approach involves building a ResNet50 model from scratch and fine-tuning it to classify these dental diseases.

## Overview

1. **Data Pre-Processing:**
   - Normalization and augmentation are performed to prepare the dental images for analysis. This ensures the images are in optimal condition for model training and evaluation.
   - Visualize the distribution of the classes to understand the balance of the dataset.
   - Display images before and after augmentation to evaluate the effectiveness of preprocessing techniques and ensure the transformations are enhancing the dataset appropriately.

2. **Class Imbalance:**
   - The data was found to be not totally balanced. Therefore, weights were assigned to each class to address this imbalance.

3. **Model Building:**
   - **Identity Function:** Preserves the spatial dimensions while mapping the input to the output.
   - **Convolutional Block:** Changes the spatial dimensions while having a shortcut connection that undergoes its own convolutional transformations to match the dimensions of its path.
   - **ResNet50 Function:** Includes a number of transformations such as identity blocks, convolutional blocks, etc., and is fine-tuned with an output layer having 7 units and a 'softmax' activation function.

4. **Training Techniques:**
   - Implemented techniques like early stopping and ReduceLROnPlateau to improve the model's performance.
   - Trained the model with 250 epochs, achieving great validation accuracy evaluation accuracy.


