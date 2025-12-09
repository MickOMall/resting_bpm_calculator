# Resting Heart Rate Predictor

**Authors:** Isaac Schaffer-Neitz, Michael O’Malley  

## Overview
The Resting Heart Rate Predictor is a machine learning-based tool that estimates an individual’s resting heart rate (BPM) using common health and lifestyle metrics. It combines predictive modeling, explainability, and a user-friendly interface to provide actionable cardiovascular insights.  

The system is built with a **Random Forest Regressor**, a **Streamlit** web interface, and a **CrewAI-powered explanation layer** that provides clear, natural-language reasoning behind each prediction. Users can also calculate utilities like BMI and provide feedback to improve the model over time.  

## Features
- Predict resting heart rate using lifestyle and physical metrics
- Feature-level explanations in plain language
- Interactive BMI and other health calculators
- Feedback system for user-submitted true BPM values
- Simple, intuitive web interface  

## System Design
- **Data Processing:** One-hot encoding for categorical variables, normalization of numeric features.  
- **Model:** Random Forest Regressor trained on tabular health data, evaluated via R², MAE, and MAPE.  
- **Explainability:** CrewAI interprets feature contributions and generates human-readable explanations.  
- **Web App:** Streamlit interface for input, prediction, visualization, and feedback submission.  

## Requirements
- Python 3.9+  
- pandas  
- numpy  
- scikit-learn  
- streamlit  
- joblib  
- treeinterpreter  
- crewai  

## Use instructions :
1. Download the zip.
2. Run the code in Requirements.txt
3. Run all cells in model.ipynb to get the model.
4. Go to app.py's directory and run "streamlit run app.py"
