# Predicting End Dates with Random Forest

## Overview
This project uses a **Random Forest Regressor** to predict missing end dates in a real dataset of **road construction projects** from the **Central Bureau of Statistics, Israel** (data use permitted).  
The dataset contains various details about roadworks, but some rows (projects) are missing their completion dates.  
The goal is to estimate these missing dates as accurately as possible using machine learning.

## How It Works
1. **Data Preprocessing**  
   - Load and clean the original dataset.  
   - Remove irrelevant columns.  
   - Add **cyclical features** for months and one-hot encode categorical variables.  

2. **Feature Engineering**  
   - Encode months and years as both cyclical variables and categories.  
   - Generate additional date-related features to improve predictions.  

3. **Model Training**  
   - Train a **Random Forest Regressor** on projects with known start and end dates.  
   - Evaluate the model using **MAE**, **RMSE**, and **R²** metrics.  
   - Select the top N most important features for a more focused second training pass.  

4. **Prediction**  
   - Apply the trained model to rows with missing end dates.  
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
- Libraries:  
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
```
4. The script will:
* Train the model.
* Evaluate its performance.
* Predict missing end dates and print them

## Example Output
```sql
--- Evaluation with all features ---
MAE:  150.23
RMSE: 210.87
R²:   0.65

Predictions for Missing End Dates:
   predicted gap  predicted end year  predicted end month
0     320.123456                2020                    4
1     185.987654                2019                    9
```

## Next Steps
* Improve feature engineering for temporal data.
* Try boosting-based models (e.g., XGBoost, LightGBM) for better performance.
* Incorporate external datasets (e.g., weather, regional workload) to enrich predictions.

## Data Source & License
This project uses a roadworks dataset from the Central Bureau of Statistics (Israel).
Data use is permitted under the CBS terms. Raw data is **not redistributed** in this repo.
If you have access to the dataset, place it at `data/roadworks_cbs_israel.xlsx`.
Otherwise, use the provided synthetic sample in `examples/`.

Attribution: © Central Bureau of Statistics, Israel. All rights reserved by the respective owner.

## Ethical Use & Limitations
Predictions are estimates with a non-trivial error margin and can be wrong for specific projects.
They should not be used as the sole basis for operational, legal, or financial decisions.
The model currently serves research/illustrative purposes and is under active improvement.
