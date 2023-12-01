# Data Analytics, Assignment 4: ETL + Training a model to predict pricing and sales
**Alexandra Zana**
***40131077***
-------
**Assignment requirements:**

Question 1. [2.5 points] Load oil.csv. This file contains years worth of data of the daily
oil price. However, the data is missing for a few days. Make sure that every day contains
a value using any data imputation technique that you learned during the data preparation
week or during the missing values imputation week.

Question 2. [2.5 points] Augment the data in test.csv and train.csv with the oil price.

Question 3. [2.5 points] Note that the training set contains a ‘sales’ column while the test
set does not. Use the training set to train a model of your choice and use that model to
predict which values for sales should be in the test set. You should try training at least 2
models and compare their accuracy later.

Question 4. [2.5 points] Compare your prediction with the prediction found in submis-
sion.csv with 3 different methods:
(a) Root Mean Square Error (RMSE)
(b) Mean absolute deviation
(c) Another metric of your choice Compare the three errors. Are they in agreement? Do you
think any of the methods is objectively better than the others in this case?

Note: The data files are too large to upload

All solutions are found in the `tasks.ipynb` file.

--------
**Question 4 Analysis**

All three metrics show a consistent trend: the Random Forest model has significantly lower errors compared to the Linear Regression model. This consistency across different types of error metrics indicates a strong agreement in their assessment of the models' performance.

The RFM is a better model for this case because it has lower errors across all three metrics (RMSE, MAD, and MedAD), indicating that it is better at predicting the actual values of the data points.

A lower RMSE indicates that it is better at handling larger deviations in predictions.
A lower MAD indicates that, on average, it makes smaller errors than the Linear Regression model.
A lower MedAD, which indicates that the median size of errors is much smaller, indicating more consistent and accurate predictions for the majority of the data points.

Interpretation of the errors:

RMSE: It emphasizes larger errors more due to the squaring of the residuals. The fact that RF has a substantially lower RMSE suggests it's better at handling larger deviations in predictions.
MAD: This metric gives an average level of error and is less sensitive to outliers than RMSE. The lower MAD for the Random Forest model indicates that on average, it makes smaller errors than the Linear Regression model.
MedAD: This metric gives the typical or median error in the predictions and is robust to outliers. The significantly lower MedAD for the Random Forest model suggests that the median size of errors is much smaller, indicating more consistent and accurate predictions for the majority of the data points.
