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

## Running the Code
### Prerequisites
Ensure you have **Python 3.x** installed with the following dependencies:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn keras tensorflow
```

### Instructions
1. **Download the dataset** (`stroke_data.csv`) and place it in the same directory as the script.
2. **Run the script** using:
   ```bash
   python stroke_prediction.py
   ```
3. The model will output **error metrics (MSE, MAE)** and display feature selection results.

## Output Files
- **Trained deep learning model for stroke prediction.**
- **Feature importance visualization (bar chart, heatmap).**

## Conclusion
This project explores deep learning techniques to classify stroke occurrence using medical and lifestyle features. The convolutional layers enhance feature extraction, while proper feature selection improves model efficiency.
