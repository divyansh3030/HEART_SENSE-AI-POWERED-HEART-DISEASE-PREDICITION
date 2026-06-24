# HeartSense

❤️ HeartSense – AI-Powered Heart Disease Prediction System
Overview

HeartSense is a full-stack web application that predicts the likelihood of heart disease using Machine Learning and Deep Learning models. The system allows users to create accounts, perform heart disease risk assessments, store prediction records, and monitor patient history through an interactive dashboard.

The project combines Artificial Intelligence, Flask, MySQL, and modern web technologies to provide an intelligent healthcare support system..

Disclaimer: This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

Features
🔐 User Authentication
User Registration
Secure Login & Logout
Password Hashing
Session Management
❤️ Heart Disease Prediction
Predicts heart disease risk using clinical parameters
Probability-based risk assessment
Disease and No-Disease percentage scores
Risk categorization
📊 Dashboard Analytics
Total Predictions
Low Risk Patients
Moderate Risk Patients
High Risk Patients
Very High Risk Patients
Risk Distribution Statistics
Prediction Trends
🗂 Prediction Management
Save Patient Records
View Prediction History
Search Patients
Filter by Risk Level
Delete Records
Detailed Prediction Reports
👤 Profile Management
Update Name
Update Email
Change Password
Account Settings
🔌 REST API
API endpoint for prediction requests
JSON response support
Technology Stack
Frontend
HTML5
CSS3
Jinja2 Templates
Backend
Python
Flask
Database
MySQL
Machine Learning
Scikit-Learn
Random Forest Classifier
StandardScaler
Cross Validation
Deep Learning
TensorFlow
Keras
Talos Hyperparameter Optimization
Libraries
Pandas
NumPy
Joblib
Werkzeug
Dataset

The project uses the UCI Heart Disease Dataset.

Dataset Information
Total Records: 1025
Features: 13 Clinical Attributes
Target Classes:
0 = Heart Disease Present
1 = No Heart Disease
Input Features
Feature	Description
age	Age of patient
sex	Gender
cp	Chest pain type
trestbps	Resting blood pressure
chol	Cholesterol level
fbs	Fasting blood sugar
restecg	Resting ECG results
thalach	Maximum heart rate achieved
exang	Exercise induced angina
oldpeak	ST depression
slope	Slope of peak exercise ST segment
ca	Number of major vessels
thal	Thalassemia status
Machine Learning Pipeline
Data Preprocessing
Data Cleaning
Feature Selection
Standard Scaling
Train-Test Split
Stratified Sampling
Random Forest Model
100 Decision Trees
Balanced Class Weights
Cross Validation
Feature Importance Analysis
Deep Neural Network
Multiple Dense Layers
ReLU Activation
Dropout Regularization
Sigmoid Output Layer
Adam Optimizer
Hyperparameter Tuning using Talos
Database Design
Users Table

Stores registered user information.

Fields:

id
fullname
email
password_hash
created_at
Predictions Table

Stores prediction history.

Fields:

patient_name
clinical inputs
prediction result
disease probability
no disease probability
risk level
created_at
Risk Categories
Probability	Risk Level
< 30%	Low
30% – 60%	Moderate
60% – 80%	High
> 80%	Very High

## Setup
1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install the app requirements.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Create the MySQL database and tables from [schema_mysql.sql](E:/heartsense/schema_mysql.sql).

4. Optional: set custom environment variables if you do not want to use the defaults in [app.py](E:/heartsense/app.py).

```powershell
$env:MYSQL_HOST="127.0.0.1"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your-password"
$env:MYSQL_DATABASE="heartsense"
$env:FLASK_SECRET_KEY="replace-this"
```

5. Run the Flask app.

```powershell
python app.py
```

6. Open the app in your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Model Files
The app expects these saved files inside [models](E:/heartsense/models):
- `heartsense_pipeline.pkl`
- `heartsense_dnn.h5`
- `dnn_scaler.pkl`

Train Models
Random Forest
python train_model.py
Deep Neural Network
python train_dnn_talos.py

If you need to retrain the scikit-learn pipeline:

```powershell
python train_model.py
```

The optional research script [train_dnn_talos.py](E:/heartsense/train_dnn_talos.py) is not required to run the Flask app and may need extra package compatibility work if you want to use it.

## Notes
- This project is for educational and demonstration use only, not medical advice.
- Prediction inputs are validated before inference and saved per logged-in user.
- If you switch Python environments, keep `scikit-learn==1.3.2` to stay compatible with the saved pipeline model.

Future Enhancements
Email Reports
PDF Report Generation
Doctor Dashboard
Multi-user Roles
Real-time Health Monitoring
Cloud Deployment (AWS)
Model Explainability (SHAP/LIME)
Mobile Application
Author

Divyansh Kakkar

B.Tech Computer Science Engineering
Galgotias University

AI • Machine Learning • Full Stack Development • Cloud Computing