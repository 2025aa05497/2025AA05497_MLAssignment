# 📊 Machine Learning Assignment 2 – Classification Models & Deployment

## 1. Problem Statement
This project implements and compares multiple machine learning classification models on the Bank Marketing Dataset to predict whether a customer will subscribe to a term deposit.

The project demonstrates a complete end-to-end machine learning pipeline:

* Data preprocessing
* Feature encoding
* Model training
* Model evaluation
* Model comparison
* Model deployment using Streamlit

## 2. Dataset Description
- **Dataset Name:** Bank Marketing Dataset
- **Source:** UCI Machine Learning Repository
- **Problem Type:** Binary Classification(0/1)
- **Number of Instances:** 45211
- **Number of Features:** 16 input features
- **Target Variable:**  
  - `target = 1` → Customer subscribed
  - `target = 0` → Customer did not subscribe
- **Description:**  
  The dataset contains demographic details, campaign-related information, and economic attributes collected from direct marketing campaigns of a Portuguese banking institution.
The dataset includes both categorical and numerical features. Categorical variables were converted using One-Hot Encoding before training the models.
---

## 3. Models Used and Evaluation Metrics

### Implemented Models
1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Metrics
The following evaluation metrics were used:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|---------|-----|----------|--------|----|-----|
| Logistic Regression | 0.9015 | 0.9060 | 0.6441 | 0.3533 | 0.4563 | 0.4295 |
| Decision Tree | 0.8759|0.6958	|0.4688	|0.4607|	0.4647	|0.3945 |
| kNN | 0.8938|	0.8099|	0.5845	|0.3192	|0.4129|	0.3800
| Naive Bayes | 0.8638	|0.8227|	0.4309|	0.5144	|0.4690|	0.3935
| Random Forest(Ensemble) | 0.9058	|0.9264	|0.6671	|0.3880	|0.4907|	0.4625
| XGBoost(Ensemble) | 0.9081	|0.9314|	0.6557	|0.4508|	0.5343|	0.4957

---

## 4. Model Observations

| ML Model Name| Observation about model performance |
|------|------------|
| Logistic Regression | Provides stable baseline performance with good AUC but relatively low recall. |
| Decision Tree | Moderate performance with lower AUC; slightly prone to overfitting. |
| KNN | Sensitive to feature scaling; lower recall compared to other models. |
| Naive Bayes |Achieves higher recall among simple models but lower precision.|
| Random Forest(Ensemble) |Strong overall performance with good generalization and balanced metrics. |
| XGBoost (Ensemble)| Best overall model with highest Accuracy, AUC, F1 Score, and MCC. |

---

## 5. Streamlit Application
The Streamlit app allows:
- Upload custom test CSV file
- Select a classification model
- Generate predictions instantly
- View evaluation metrics
- Display confusion matrix
- View classification report
- Download sample test dataset

---

## 6. Deployment
The application is deployed using Streamlit Community Cloud and can be accessed via a public deployment link.

---

## 7. Compliance
All assignment requirements are satisfied, including:
- Implementation of six classification models
- Use of ensemble models (Random Forest and XGBoost)
- Evaluation using multiple performance metrics
- Comparative model analysis
- Deployment using Streamlit
