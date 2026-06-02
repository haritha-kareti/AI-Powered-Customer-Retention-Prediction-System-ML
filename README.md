````html
<div align="center">

# 🤖 AI-Powered Customer Retention Prediction System

### Machine Learning-Based Customer Churn Prediction

<img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Flask-Web_App-green?style=for-the-badge&logo=flask">
<img src="https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">

</div>

---

## 📌 Project Overview

This project is an **AI-Powered Customer Retention Prediction System** designed to identify customers who are likely to churn. The system uses Machine Learning techniques and a complete data preprocessing pipeline to analyze customer behavior and generate accurate churn predictions.

### 🎯 Objectives

✔ Predict customer churn

✔ Improve customer retention strategies

✔ Automate data preprocessing

✔ Handle missing values and outliers

✔ Balance imbalanced datasets using SMOTE

✔ Build robust machine learning models

---

## 🚀 Key Features

<table>
<tr>
<td>✅ Missing Value Imputation</td>
<td>✅ Outlier Treatment</td>
</tr>

<tr>
<td>✅ Feature Selection</td>
<td>✅ Categorical Encoding</td>
</tr>

<tr>
<td>✅ SMOTE Oversampling</td>
<td>✅ Feature Scaling</td>
</tr>

<tr>
<td>✅ Model Training</td>
<td>✅ Churn Prediction</td>
</tr>

<tr>
<td>✅ Logging System</td>
<td>✅ Automated ML Pipeline</td>
</tr>
</table>

---

## 🛠️ Technology Stack

<table>
<tr>
<th>Category</th>
<th>Technology</th>
</tr>

<tr>
<td>Programming Language</td>
<td>Python</td>
</tr>

<tr>
<td>Data Analysis</td>
<td>Pandas, NumPy</td>
</tr>

<tr>
<td>Visualization</td>
<td>Matplotlib, Seaborn</td>
</tr>

<tr>
<td>Machine Learning</td>
<td>Scikit-Learn</td>
</tr>

<tr>
<td>Imbalanced Data Handling</td>
<td>SMOTE</td>
</tr>

<tr>
<td>Logging</td>
<td>Python Logging Module</td>
</tr>
</table>

---

## 📂 Project Structure

```text
AI-Powered-Customer-Retention-Prediction-System/
│
├── main.py
├── logging_code.py
│
├── MICE_Imputation.py
├── variable_outliers.py
├── filter_methods.py
├── categorical_to_num.py
├── feature_scaling.py
│
├── Telco-Customer-Churn.csv
│
├── logs/
│
└── README.md
````

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Loading

```python
pd.read_csv('Telco-Customer-Churn.csv')
```

Loads customer data into a Pandas DataFrame.

---

### 2️⃣ Train-Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

* 80% Training Data
* 20% Testing Data

---

### 3️⃣ Missing Value Handling

Module:

```python
handle_missing_value()
```

Technique:

* MICE Imputation
* Numerical Data Cleaning
* Missing Value Replacement

---

### 4️⃣ Data Separation

Numerical Features:

```python
select_dtypes(exclude='object')
```

Categorical Features:

```python
select_dtypes(include='object')
```

---

### 5️⃣ Outlier Treatment

Module:

```python
vt_outliers()
```

Purpose:

* Reduce skewness
* Improve model performance
* Stabilize feature distributions

---

### 6️⃣ Feature Selection

Module:

```python
fm()
```

Benefits:

* Removes irrelevant features
* Improves accuracy
* Faster training

---

### 7️⃣ Categorical Encoding

Module:

```python
c_t_n()
```

Converts categorical values into machine-readable numerical values.

---

### 8️⃣ Data Balancing

```python
SMOTE(random_state=42)
```

Balances minority and majority classes.

Benefits:

* Prevents model bias
* Improves churn prediction accuracy

---

### 9️⃣ Feature Scaling

Module:

```python
fs()
```

Standardizes numerical features before model training.

---

## 📊 Pipeline Architecture

```text
Dataset
   │
   ▼
Missing Value Handling
   │
   ▼
Data Separation
   │
   ▼
Outlier Treatment
   │
   ▼
Feature Selection
   │
   ▼
Categorical Encoding
   │
   ▼
SMOTE Balancing
   │
   ▼
Feature Scaling
   │
   ▼
Model Training
   │
   ▼
Customer Churn Prediction
```

---

## 📈 Dataset Information

### Dataset

**Telco Customer Churn Dataset**

### Target Variable

```text
Churn
```

Classes:

```text
Yes → Customer Leaves

No → Customer Stays
```

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/haritha-kareti/AI-Powered-Customer-Retention-Prediction-System-ML.git
```

### Move Into Project

```bash
cd AI-Powered-Customer-Retention-Prediction-System-ML
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Project

```bash
python main.py
```

---

## 📷 Screenshots

<div align="center">

### Home Page

<img src="images/home.png" width="800">

### Prediction Result

<img src="images/result.png" width="800">

</div>

---

## 🎯 Business Benefits

* Reduce Customer Churn
* Improve Customer Retention
* Increase Revenue
* Improve Customer Satisfaction
* Enable Data-Driven Decisions

---

## 🔮 Future Enhancements

* Flask Web Application
* Real-Time Prediction API
* AWS Deployment
* Explainable AI (SHAP)
* Dashboard Analytics
* Automated Retraining

---

## 👨‍💻 Author

### Haritha Kareti

Machine Learning Engineer

<p>
<a href="https://github.com/haritha-kareti">
GitHub Profile
</a>
</p>

---

<div align="center">

### ⭐ Star this repository if you found it useful!

Made with ❤️ using Python & Machine Learning

</div>
```
