# Flight Fare Prediction

A machine learning project to predict flight prices based on travel details using a **Random Forest Regression** model built from scratch.  
The model is trained on a dataset of **10,000 flight records**, with categorical features handled via **One-Hot Encoding**.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Preprocessing Steps](#preprocessing-steps)
- [Model Development](#model-development)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

## Overview

This project focuses on predicting the price of flight tickets using various travel details such as date, month, day of travel, source, and destination.  
A **Random Forest Regression** model is developed from scratch without using any pre-built machine learning libraries for modeling, ensuring a deeper understanding of the algorithm's internal mechanics.

---

## Features

- ✈️ Predict flight fares based on:
  - **Date of Travel**
  - **Day of Travel**
  - **Month of Travel**
  - **Source Airport**
  - **Destination Airport**
- 🔥 One-Hot Encoding for handling categorical (text) features.
- 🌳 Custom-built **Random Forest Regressor** from scratch (no scikit-learn).
- 📈 High prediction accuracy after training on 10k+ instances.

---

## Dataset

- **Size:** 10,000+ rows
- **Features:**
  - **Date** of flight
  - **Day** of week
  - **Month** of travel
  - **Source** airport
  - **Destination** airport
  - **Flight Fare** (target variable)

> 🚀 Dataset is structured and cleaned for modeling.  

---

## Technologies Used

- **Python 3**
- **Pandas** – for data manipulation.
- **NumPy** – for numerical operations.
- **Matplotlib / Seaborn** – for optional data visualization.
- **Pure Python** – for building Random Forest algorithm.

---

## Preprocessing Steps

1. **Handling Textual Data:**
   - Applied **One-Hot Encoding** on categorical columns (Source, Destination).
   
2. **Feature Engineering:**
   - Extracted day, month, and weekday from the flight date.
   
3. **Train-Test Split:**
   - Split dataset into training and testing sets for evaluation.

---

## Model Development

- **Random Forest Regression** is implemented manually:
  - Pre Trained **Random Forest Model**.
  - Final prediction = **average of individual tree predictions**.
  - Bootstrap aggregation (bagging) is used to reduce variance.
  
- **Advantages:**
  - Handles overfitting better than single decision trees.
  - Robust to outliers and missing values.

---

## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/gsphanitalpak/IDS.git
```

2. **Install Required Libraries**

```bash
pip install -r requirements.txt
```

3. **Place the Dataset**

Data should be in CSV file format with features as:
  - **Date** of flight
  - **Day** of week
  - **Month** of travel
  - **Source** airport
  - **Destination** airport
  - **Flight Fare** (target variable)

4. **Run the Project**

```bash
python flight_fare_prediction.py
```

---

## Usage

- Train the Random Forest model on the provided dataset.
- Predict the price of flights based on travel details.
- Evaluate the model's performance using RMSE, MAE, and R² score.

Example Prediction:

```python
sample_input = {
    'day': 5,
    'month': 12,
    'source': 'New Delhi',
    'destination': 'Mumbai'
}
predicted_fare = model.predict(sample_input)
print(f"Predicted Flight Fare: ₹{predicted_fare}")
```

---

## Future Improvements

- 📈 Hyperparameter tuning (e.g., number of trees, max depth).
- 🛡️ Add model validation (Cross-validation).
- 🌍 Expand dataset to include airline companies and flight durations.
- 📊 Deploy a simple web app (Flask/Django) for live predictions.
- 🧠 Explore advanced algorithms like XGBoost or CatBoost.

---
## Contact
Created and maintained by **Santhosh Phanitalpak Gandhala (https://github.com/gsphanitalpak)**. 
For any questions, feel free to reach out!
