📊 PCA Implementation with Visualization

This project demonstrates a simple and intuitive implementation of Principal Component Analysis (PCA) along with visualization and classification using Logistic Regression.

The goal is to show how high-dimensional data can be reduced to fewer dimensions while retaining most of the important information, and how this transformation affects model performance and data visualization.

🚀 Features
📌 Custom dataset with features: Height, Weight, Age
⚖️ Feature standardization using StandardScaler
🔽 Dimensionality reduction using PCA
🤖 Classification using Logistic Regression
📉 Confusion Matrix for evaluation
🎨 Visual comparison before and after PCA
🧠 What is PCA?

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data into a new coordinate system. The new axes (principal components) capture the maximum variance in the data with fewer dimensions.

📂 Project Workflow
1️⃣ Import Libraries

We use:

numpy, pandas → Data handling
scikit-learn → PCA, scaling, model training
matplotlib, seaborn → Visualization
2️⃣ Dataset Creation

A small synthetic dataset is created with:

Height
Weight
Age
Gender (Target: Male = 1, Female = 0)
3️⃣ Data Preprocessing
Features are separated from the target
Standardization is applied so all features have:
Mean = 0
Standard Deviation = 1
4️⃣ Apply PCA + Model Training
Reduce dimensions from 3 → 2
Split data into:
70% Training
30% Testing
Train a Logistic Regression model
Predict gender labels
5️⃣ Model Evaluation
Confusion Matrix is used to evaluate predictions
Visualized using a heatmap for clarity
6️⃣ Visualization

Two plots are generated:

Before PCA
Uses first two standardized features
After PCA
Uses principal components

This helps visualize how PCA transforms the data.

<img width="1026" height="419" alt="image" src="https://github.com/user-attachments/assets/12f00280-8eba-40d3-868d-26a3640a644a" />
