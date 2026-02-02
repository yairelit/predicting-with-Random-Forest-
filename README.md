# Predicting End Dates with Random Forest

## Overview
This project uses a **Random Forest Regressor** to predict missing end dates in a real dataset of **road construction projects** from the **Central Bureau of Statistics, Israel** (data use permitted).  
The dataset contains various details about roadworks, but some rows (projects) are missing their completion dates.  
The goal is to estimate these missing dates as accurately as possible using machine learning.

## Dataset Note
> [!IMPORTANT]
> The data file included in this repository is a **small sample** of the original dataset. The full dataset consists of **tens of thousands of records** and is not included here due to its large size. The logic and model training processes are designed to scale and remain fully compatible with the complete data structure.

## How It Works
1. **Data Preprocessing** - Load and clean the original dataset.  
   - Remove irrelevant columns.  
   - Add **cyclical features** for months and one-hot encode categorical variables.  

2. **Feature Engineering** - Encode months and years as both cyclical variables and categories.  
   - Generate additional date-related features to improve predictions.  

3. **Model Training** - Train a **Random Forest Regressor** on projects with known start and end dates.  
   - Evaluate the model using **MAE**, **RMSE**, and **R²** metrics.  
   - Select the top N most important features for a more focused second training pass.  

4. **Prediction** - Apply the trained model to rows with missing end dates.  
   - Calculate the predicted duration (`gap`) and reconstruct the predicted end date, year, and month.

## Current Performance
The current model produces **meaningful predictions**, but the **error margin is relatively large**.  
This is likely due to the inherent difficulty of predicting project completion times based solely on the available features.  
Ongoing work is being done to improve accuracy by:
- Exploring alternative feature engineering strategies.
- Testing other algorithms in addition to Random Forest.
- Fine-tuning hyperparameters.

## Limitations
- The predictions may be **inaccurate for individual cases**, especially for atypical projects.  
- Dataset characteristics make exact prediction inherently challenging.

## Requirements
- Python 3.8+
  
Libraries:  
- pandas
- numpy
- matplotlib
- scikit-learn
- openpyxl
- xlrd  


## Running the Project
1. Place the dataset in the project directory.
2. Update the `FILE_PATH` variable in the script to point to your dataset.
3. Run:
 ```bash
 python main.py
