#  Student Performance Predictor

A Machine Learning web application built using Python and Streamlit to predict students' final academic scores based on various factors such as study hours, attendance, previous scores, sleep hours, and parental support.

##  Project Overview

The Student Performance Predictor uses a Random Forest Regression model to estimate a student's final score based on academic and personal attributes.

The application provides an interactive interface where users can enter student details and receive predicted performance results.

##  Features

- Interactive Streamlit web interface
- Student performance prediction
- Random Forest Regression model
- Data preprocessing and missing-value handling
- Categorical feature encoding using One-Hot Encoding
- Model evaluation using MAE and R² score
- Student performance categorization

##  Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Random Forest Regression

##  Dataset Information

The dataset contains 1,500 student records with 11 columns, including:

- Age
- Gender
- Study Hours
- Attendance
- Previous Score
- Sleep Hours
- Assignments
- Extracurricular Activities
- Parental Support
- Internet Access
- Final Score (Target Variable)

##  Machine Learning Model

The project uses Random Forest Regression to predict students' final scores.

The preprocessing pipeline includes:
- Missing value imputation
- One-Hot Encoding for categorical variables
- Feature preprocessing using ColumnTransformer
- Model training and testing

## 📈 Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- R² Score



