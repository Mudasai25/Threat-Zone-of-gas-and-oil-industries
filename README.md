# Threat Zone Prediction in Oil and Gas Refineries

This repository contains the source code and documentation for the project **"Threat Zone of an Explosion Particularly in Oil and Gas Industries or Refineries."** The project is designed to assess and predict explosion risks in oil and gas refineries using machine learning, real-time weather data, and refinery-specific characteristics.

---

## 📖 Overview

Explosions in oil and gas industries can cause catastrophic damage, including harm to people, infrastructure, and the environment. This project introduces a web-based application for predicting and analyzing explosion threat zones by leveraging:

- **Machine learning algorithms** (e.g., Random Forest Classifier)
- **Real-time weather data** (via OpenWeatherMap API)
- **Industry-specific datasets** (latitude, longitude, gas types, and more)

---

## 🌟 Key Features

### 1. Gas Classification
- Utilizes machine learning to classify gases based on explosion risk.
- Predicts threat levels using features such as wind speed, cloud cover, and insolation.

### 2. Weather Integration
- Fetches real-time weather data using OpenWeatherMap API:
  - Wind speed
  - Cloud cover
  - Solar insolation (radiation)

### 3. Risk Prediction
- Calculates explosion risk zones using mathematical models for insolation, stability classes, and meteorological parameters.

### 4. Interactive Web Application
- Built with Streamlit for an intuitive user interface.
- Allows users to upload refinery data, view predictions, and analyze threat zones interactively.

---

## 🛠️ Tools and Technologies

### Programming Language:
- Python

### Libraries:
- TensorFlow
- Random Forest Classifier
- Streamlit
- NumPy
- Pandas

### APIs:
- OpenWeatherMap API (for real-time weather data)

### Mathematical Models:
- Solar radiation calculations
- Stability class computations

---

## 🧪 Methodology

### 1. Data Collection
- Gathered refinery data (latitude, longitude, gas types).
- Preprocessed the data and handled missing values.

### 2. Model Training
- Trained a Random Forest Classifier to classify gases and predict explosion risks.
- Applied data augmentation to enhance dataset diversity.

### 3. Web Application
- Integrated with OpenWeatherMap API for weather data.
- Developed interactive maps to display threat zones and their geographical impact.

---

## 🧮 Mathematical Models

### Daytime Insolation (Q):
Q = S ⋅ (r/d)^2 ⋅ cos(zenith angle)

### Evaluation Metrics:
- Accuracy:  
  Accuracy = (TP + TN) / Total Predictions

- Precision:  
  Precision = TP / (TP + FP)

- Recall:  
  Recall = TP / (TP + FN)

- F1 Score:  
  F1 Score = 2 ⋅ (Precision ⋅ Recall) / (Precision + Recall)

---

## 📊 Results
- Achieved **92% accuracy** during model evaluation.
- Demonstrated potential in identifying threat zones, though further improvements are needed to address dataset imbalances.

---

## 🚀 How to Use

### 1. Clone the repository:
   git clone https://github.com/Mudasai25/BreadcrumbsThreat-Zone-of-gas-and-oil-industries.git

### 2. Install dependencies:
   pip install -r requirements.txt

### 3. Run the Streamlit application:
   streamlit run app.py 

### 4. Upload refinery data:
- Add industry-specific data (latitude, longitude, gas type, etc.).
- View explosion risk predictions and threat zone visualizations.

---

## 📚 References
- OpenWeatherMap API
- Research papers and relevant studies on explosion risk modeling

---

## 📄 Future Work
- Enhance the dataset with more diverse and balanced samples.
- Integrate advanced machine learning techniques for improved risk prediction.
- Add features such as:
  - Evacuation plan mapping
  - Emergency response recommendations
