# Stroke Prediction Using Deep Neural Networks

## Overview
This project implements a **Deep Neural Network (DNN)** to predict stroke occurrence based on medical and lifestyle factors. It applies **feature selection techniques** and **convolutional layers** to enhance predictive performance. The model is trained and evaluated using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.

## Dataset
- The dataset is stored in **stroke_data.csv**.
- It contains medical and demographic features, including:
  - **Age, Gender, Hypertension, Heart Disease, BMI, Average Glucose Level, Smoking Status, Socioeconomic Status (SES), and Stroke Occurrence (target variable).**

## Implementation Details

### **1. Data Preprocessing**
- **Encoding categorical variables:**
  - Binary categorical variables (e.g., Gender) are **label-encoded**.
  - Multi-category variables (e.g., Smoking Status, SES) are **one-hot encoded**.
- **Feature scaling:**  
  - Standardized using **StandardScaler** to normalize numerical features.

### **2. Feature Selection**
- **Correlation Analysis:**  
  - A heatmap visualizes correlations between features and stroke occurrence.
- **Gini Impurity (Decision Tree Regressor):**  
  - Important features are ranked based on their impact.
- **Variance Thresholding:**  
  - Removes low-variance features to improve model efficiency.

### **3. Deep Learning Model Architecture**
- **Convolutional Layers (Conv1D, MaxPooling1D)** extract patterns from feature space.
- **Global Max Pooling Layer** reduces feature dimensions.
- **Dense Layers** with ReLU activation function for classification.
- **Dropout Layer** prevents overfitting.
- **Adam Optimizer** used for training.
- **Loss function:** Mean Squared Error (MSE).

### **4. Model Training and Evaluation**
- The dataset is split into **80% training, 20% testing**.
- **100 epochs, batch size = 50.**
- Performance metrics:
  - **Mean Squared Error (MSE)**
  - **Mean Absolute Error (MAE)**

## Output Files
- **Trained deep learning model for stroke prediction.**
- **Feature importance visualization (bar chart, heatmap).**
