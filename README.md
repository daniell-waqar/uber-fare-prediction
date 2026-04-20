# Uber Fare Prediction NYC

A machine learning web application that predicts Uber ride fares 
in New York City using historical trip data.

## Project Overview
Trained on 190,000+ real Uber trips from the Kaggle Uber Fares Dataset. 
Achieved R² of 0.83 and RMSE of $3.50 using Gradient Boosting Regressor.

## Key Features
- Haversine formula to compute trip distance from raw GPS coordinates
- Temporal feature engineering (rush hour, night, weekend flags)
- Airport proximity detection (JFK, LGA, EWR)
- Compared Linear Regression, Random Forest, and Gradient Boosting
- Interactive fare prediction UI built with Streamlit

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib

## How to Run
pip install -r requirements.txt
streamlit run app.py

## Results
| Model              | RMSE  | MAE   | R²    |
|--------------------|-------|-------|-------|
| Linear Regression  | $3.74 | $2.12 | 0.808 |
| Random Forest      | $3.66 | $2.06 | 0.816 |
| Gradient Boosting  | $3.50 | $1.92 | 0.832 |

## Dataset
Uber Fares Dataset — https://www.kaggle.com/datasets/yasserh/uber-fares-dataset
