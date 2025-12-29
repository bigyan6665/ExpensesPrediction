# Expenses Prediction

## Overview
This project predicts **personal expenses** using demographic and lifestyle attributes. It is designed as an **end-to-end machine learning pipeline** that automates data preprocessing, model training, hyperparameter tuning, evaluation, logging, and artifact management. The system selects the best-performing model based on the R² score and exposes it for prediction through a web application interface.

---

## Problem Statement
The objective of this project is to predict individual expenses based on the following features:
- Age
- Sex
- BMI
- Number of children
- Smoker status
- Region

Accurate expense prediction can assist insurers, analysts, and individuals in cost estimation, budgeting, and risk assessment.

---

## Dataset Description
- Source: Cleaned CSV dataset (integrated by default)
- Number of records: **1,338**
- Input features:
  - `age`
  - `sex`
  - `bmi`
  - `children`
  - `smoker`
  - `region`
- Target feature:
  - `expenses`

The dataset undergoes preprocessing including encoding of categorical variables and feature transformation through a pipeline.

---

## Technical Features
- Automated ML pipeline with preprocessing and model training
- Multiple regression models trained with hyperparameter tuning
- Model selection based on **R² score**
- Centralized logging for the entire training workflow
- Versioned storage of:
  - Intermediatory data
  - Preprocessing objects
  - Trained models

---

## Usage Procedure
1. Run the application.
2. The **Home Page** is displayed.
3. Click **"Click to start training"** to initiate the full training pipeline.
4. After training completes, click **"Go to prediction page"**.
5. Enter feature values to generate expense predictions using the best-trained model.

---

## Logging and Artifacts
- All training steps, model evaluations, and errors are logged in the `logs/` directory.
- Trained models, preprocessors, and intermediate outputs are stored in the `artifacts/` directory for reproducibility and reuse.

---

## Tech Stack
- Python
- Scikit-learn
- Pandas, NumPy
- Logging module
- Flask

---

## Key Highlights
- End-to-end ML pipeline implementation
- Automated model selection
- Production-oriented logging and artifact management
- Clean, modular project structure

