# Codekshetra_Aquavision_Ai

AquaVision AI 🌍💧

AI-Powered Groundwater Forecasting & Smart Water Management System
AquaVision AI is an intelligent web-based platform that analyzes groundwater monitoring data and predicts future groundwater levels using Machine Learning. The system helps farmers, researchers, and policymakers make informed decisions about water usage and agricultural planning.

📌 Problem Statement
Groundwater is one of the most critical natural resources for agriculture and daily life. However, many regions experience unpredictable fluctuations in groundwater levels due to climate changes, excessive extraction, and poor water management.

Farmers and authorities often lack tools to predict future water availability.

AquaVision AI solves this problem by providing:
Groundwater level forecasting
Region-based insights
Data visualization dashboards
AI-powered analysis
🚀 Key Features
🔹 AI Groundwater Forecasting
Predicts future groundwater levels using historical monitoring data.

🔹 Location-based Prediction
Users can use their device location to get predictions from the nearest groundwater station.

🔹 Interactive Dashboard
Visualizes groundwater data with charts and statistics.

🔹 Data Insights
Displays:

Average water levels
Monthly groundwater trends
Basin distribution
Top monitoring stations

🔹 AI Trend Analysis
Uses AI to analyze groundwater trends and possible risks.
🧠 Machine Learning Model
The forecasting model is built using XGBoost Regressor, trained on historical groundwater monitoring data.
Model Details
Dataset rows after cleaning: 550,841

Training samples: 443,424
Features used:
Lag1
Lag2
Lag3
Lag4
Month
Year
Station Encoding
Model Performance
MAE: 2.0782
R² Score: 0.8156

This indicates strong predictive performance for groundwater forecasting.
🛠️ Technology Stack
Backend
Python
FastAPI
Pandas
NumPy
Joblib
Machine Learning
XGBoost
Scikit-learn
Frontend
HTML
CSS
JavaScript
Chart.js
APIs Used
FastAPI REST APIs
Browser Geolocation API

📊 Dashboard Visualizations
The system includes multiple visual analytics:
Average water level by state
Monthly groundwater trends
Basin distribution
Top groundwater monitoring stations
Station-level groundwater data table
