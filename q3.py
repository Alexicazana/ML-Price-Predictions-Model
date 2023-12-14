import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

########Question 4########
# Gotta reconstruct the id column on predictions so that I can merge on this column
# NOTE: assumption that the data is perfectly aligned (row wise), which is okay and works
test_augmented = pd.read_csv('test_augmented.csv')
predictions_test_augmented = pd.read_csv('predictions_test_augmented.csv')
predictions_test_augmented['id'] = test_augmented['id']

submission = pd.read_csv('submission.csv')

# Merge predictions with submission 
comparison_df = predictions_test_augmented.merge(submission, on='id')

print(comparison_df.head())

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to calculate MAPE
# def mape(y_true, y_pred): 
#     epsilon = 1e-10
#     return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
# EDIT: MAPE is not a good metric for this problem because there are many 0s in the data...
# I tried using a variable to offset and fix the division by 0 (epsilon), but that has weird behaviour (aka extremely high values, meaning  that there are instances where the actual sales (y_true) are very close to zero, which greatly inflates the percentage errors)
# So, I instead use Median Absolute Deviation (MedAD) as a third metric (see below)

def medad(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))

# Calculate MedAD for Linear Regression and Random Forest predictions
lr_medad = medad(comparison_df['sales'], comparison_df['lr_sales_prediction'])
rf_medad = medad(comparison_df['sales'], comparison_df['rf_sales_prediction'])

# Calculate metrics for Linear Regression predictions
lr_rmse = rmse(comparison_df['sales'], comparison_df['lr_sales_prediction'])
lr_mad = mean_absolute_error(comparison_df['sales'], comparison_df['lr_sales_prediction'])
# lr_mape = mape(comparison_df['sales'], comparison_df['lr_sales_prediction'])

# Calculate metrics for Random Forest predictions
rf_rmse = rmse(comparison_df['sales'], comparison_df['rf_sales_prediction'])
rf_mad = mean_absolute_error(comparison_df['sales'], comparison_df['rf_sales_prediction'])
# rf_mape = mape(comparison_df['sales'], comparison_df['rf_sales_prediction'])

# Print results to console 
print(f"Linear Regression - RMSE: {lr_rmse}, MAD: {lr_mad}")
print(f"Random Forest - RMSE: {rf_rmse}, MAD: {rf_mad}")
print(f"Linear Regression - MedAD: {lr_medad}")
print(f"Random Forest - MedAD: {rf_medad}")
