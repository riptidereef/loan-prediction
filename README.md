# Loan Approval Prediction

This project uses various machine learning algorithms to predict loan approval based on a user's financial profile and demographic information.

## Overview

The model takes in user-inputted financial data, processes and scales it, and then evaluates it using multiple machine learning models. It compares performance across models and uses ensemble methods for improved prediction.

## Features

- User input for real-time loan prediction
- Preprocessing and feature engineering
- Option to drop highly correlated features
- Class balancing using SMOTE
- Comparison across multiple classifiers
- Ensemble techniques: Averaging, Stacking, Bagging, Boosting
- Neural network model using Keras

## Models Used

- K-Nearest Neighbors (KNN)
- Decision Tree
- Logistic Regression
- Support Vector Machine (SVM)
- Ensemble (Averaging, Stacking, Bagging, AdaBoost)
- Neural Network (Sequential model)

## Dataset

The dataset used is `loan_approval_dataset.csv`, containing financial and demographic data. Features include:

- Loan amount, term
- Number of dependents
- Education and self-employment status
- Annual income
- Credit score
- Asset values (residential, commercial, luxury, bank)

All currency values are converted from INR to USD.

## Preprocessing Steps

1. Clean column names
2. Encode categorical variables (education, employment, status)
3. Scale features using `StandardScaler`
4. Optionally drop highly correlated columns (based on correlation heatmap)
5. Balance dataset using SMOTE

## Usage

Run the notebook and follow the prompts to input your data:

```python
loan_amount = float(input("Loan amount in USD: "))
loan_term = float(input("Loan term in years: "))
num_dependents = float(input("Number of dependents: "))
```
# ...

## Model Output

After input, each model outputs a prediction like the following:
...
Loan Approved with Logistic Regression Classifier!
Loan Rejected with SVM Classifier!
...


---

## Model Performance

| Model                | Accuracy | F1 Score |
|----------------------|----------|----------|
| Decision Tree         | 0.9690   | 0.9690   |
| Boosting (AdaBoost)   | 0.9690   | 0.9707   |
| Stacking              | 0.9671   | 0.9683   |
| Neural Network        | 0.9652   | 0.9665   |
| Bagging               | 0.9595   | 0.9623   |
| Averaging Ensemble    | 0.9417   | 0.9445   |
| Support Vector Machine (SVM) | 0.9398 | 0.9397 |
| Logistic Regression   | 0.9266   | 0.9266   |
| K-Nearest Neighbors (KNN) | 0.9078 | 0.9074 |

---

## Ensemble Techniques

- **Averaging**: Averages predictions of base models
- **Stacking**: Combines multiple classifiers with a meta-model
- **Bagging**: Uses bootstrapped training sets for variance reduction
- **Boosting (AdaBoost)**: Focuses on correcting errors of weak learners

---

## Neural Network Architecture

- 3 hidden layers: 64, 32, and 16 neurons
- ReLU activation for hidden layers
- Sigmoid activation for the output layer
- Optimizer: Adam
- Loss: Binary Crossentropy
- Accuracy achieved: ~99% training, ~96.5% test

---

## Dependencies

Install all required packages using:

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib tensorflow

