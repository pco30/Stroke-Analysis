# Stroke Prediction Using N-Fold Cross Validation

## Overview
This project applies **N-Fold Cross Validation** to train a **Deep Neural Network (DNN)** for predicting stroke occurrence. The model uses **Convolutional Neural Networks (CNNs)** to extract patterns from medical and demographic data.

## Dataset
- The dataset is stored in **stroke_data.csv**.
- It includes medical and lifestyle features such as:
  - **Age, Gender, Hypertension, Heart Disease, BMI, Avg. Glucose Level, SES, Smoking Status, and Stroke Occurrence.**

## Implementation Details

### **1. Data Preprocessing**
- **Encoding categorical variables:**
  - Binary categorical variables (e.g., Gender) are **label-encoded**.
  - Multi-category variables (e.g., Smoking Status, SES) are **one-hot encoded**.
- **Feature scaling:**  
  - Standardized using **StandardScaler** to normalize numerical features.
- **Reshaped input data** for **Conv1D layers**, treating each feature as a time step.

### **2. Model Architecture**
- **Convolutional Layers (Conv1D, MaxPooling1D)** to capture feature dependencies.
- **Global Max Pooling Layer** reduces feature dimensions.
- **Fully Connected Dense Layers** with ReLU activation.
- **Dropout Layer** prevents overfitting.
- **Output Layer** with **sigmoid activation** for binary classification.
- **Loss Function:** Binary Cross-Entropy.
- **Optimizer:** Adam.

### **3. N-Fold Cross Validation**
- **5-Fold Cross Validation** is applied using `KFold(n_splits=5)`, shuffling data before training.
- The model is trained separately for each fold, and accuracy is recorded for each validation set.
- Final performance is reported as **average accuracy across all folds**.
