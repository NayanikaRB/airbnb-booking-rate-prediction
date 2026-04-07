# Airbnb Booking Rate Prediction using Machine Learning

## Overview
This project predicts whether an Airbnb listing will achieve a high booking rate using machine learning. It combines exploratory data analysis, extensive feature engineering, and model comparison to identify the factors most associated with booking success.

## Business Problem
Understanding which listings are likely to perform well can help:
- improve listing visibility and ranking
- help hosts optimize pricing, availability, and listing quality
- support better decision-making for platform growth and revenue

## Dataset
The dataset contains Airbnb listing information, including:
- pricing and availability
- host response behavior and listing history
- property characteristics and amenities
- text-based listing information
- engineered interaction and sentiment-based features

## Project Structure

```
airbnb-booking-rate-prediction/
│
├── data/
│   ├── airbnb_train_x.csv
│   ├── airbnb_train_y.csv
│   ├── airbnb_test_x.csv
│   └── data_dictionary.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_engineering.ipynb
│   └── Models.ipynb
│
├── outputs/
│   └── high_booking_rate_predictions.csv
│
├── Report.pdf
├── README.md
└── requirements.txt
```

## Workflow

### 1. Exploratory Data Analysis
The project begins with EDA to understand distributions, missing values, feature relationships, and patterns associated with high booking rates.

### 2. Feature Engineering
A large set of engineered features was created from the raw listing data, including:
- sentiment features from listing text
- amenity and verification flags
- grouped location features
- ratio-based variables such as price per accommodate
- interaction features such as sentiment × price
- availability and booking pressure indicators

The final feature space included 144 features. :contentReference[oaicite:1]{index=1}

### 3. Model Development
Multiple models were trained and evaluated, including:
- XGBoost
- LightGBM
- Random Forest
- Gradient Boosting
- Neural Network
- Stacking ensemble
- SMOTE-based XGBoost experiments

These model families and experiments are documented in the report. :contentReference[oaicite:2]{index=2}

### 4. Final Model
The final winning model was a tuned XGBoost classifier optimized using Bayesian hyperparameter tuning with Weights & Biases. The model achieved:
- mean cross-validation AUC: 0.9147
- holdout AUC: 0.9141
- hidden test AUC: 0.916

:contentReference[oaicite:3]{index=3} 

## Key Insights
Some important findings from the analysis:
- faster host response is associated with higher booking success
- mid-sized listings that accommodate around 2 to 5 guests tend to perform best
- availability over the year is strongly related to booking performance
- cancellation policy and pricing behavior also affect booking outcomes

These insights are discussed in the project report’s EDA section. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

## Files
- `EDA.ipynb` — exploratory analysis and visual insights
- `Feature_engineering.ipynb` — feature creation and transformation logic
- `Models.ipynb` — model training, tuning, and evaluation
- `high_booking_rate_group12-10-1.csv` — prediction output / submission file
- `Data Mining Project Report.pdf` — full project documentation

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- PyTorch
- imbalanced-learn
- Weights & Biases
- Matplotlib
- Seaborn

## Dataset

The full dataset is hosted externally due to file size limitations.

Access the dataset here:
[Airbnb Booking Rate Dataset (Google Drive)](https://drive.google.com/drive/folders/1ckYff9gvIOJ0GDPt_viMSEwxGWI6DHAW?usp=sharing)

This repository contains the analysis notebooks, feature engineering pipeline, prediction outputs, and report required to reproduce the modeling workflow.

## Notes
This project was developed as part of a Data Mining and Predictive Analytics course and demonstrates a full machine learning workflow from analysis and feature engineering to model selection and evaluation.