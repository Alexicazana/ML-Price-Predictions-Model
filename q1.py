import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
# #########Q1#########
def q1(interpolated_oil_q1):
# Load the oil.csv file
    oil_df = pd.read_csv('oil.csv')

    # Convert the 'date' column to datetime and set it as index
    oil_df['date'] = pd.to_datetime(oil_df['date'])
    oil_df.set_index('date', inplace=True)

    oil_df.isna().sum()
    # Perform linear interpolation to fill missing values
    oil_df_interpolated = oil_df.interpolate(method='linear')

    # Calculate the mean of the available oil price data, excluding NaN values
    mean_oil_price = oil_df['dcoilwtico'].mean()

    # Fill initial NaN values with the calculated mean
    oil_df_interpolated['dcoilwtico'].fillna(mean_oil_price, inplace=True)

    # Create a new file 'oil2.csv' and write the interpolated data into it after filling initial NaN values with mean
    oil_df_interpolated.to_csv('oil2.csv')
    # # Displaying the first few rows after filling initial NaN values with mean
    # oil_df_interpolated.head()
    return oil_df_interpolated

#Write a function that runs the code in q1.py and returns the interpolated oil price data as a pandas DataFrame.

# #########Q2#########
# # Re-loading the train.csv and test.csv files
# train_data = 'train.csv'
# test_data = 'test.csv'

# train_df = pd.read_csv(train_data)
# test_df = pd.read_csv(test_data)

# # # Convert the 'date' column in train and test datasets to datetime
# train_df['date'] = pd.to_datetime(train_df['date'])
# test_df['date'] = pd.to_datetime(test_df['date'])

# # # Merging the train and test datasets with the oil price data
# train_augmented = train_df.merge(oil_df_interpolated, left_on='date', right_index=True, how='left')
# test_augmented = test_df.merge(oil_df_interpolated, left_on='date', right_index=True, how='left')

# # # Displaying the first few rows of the augmented datasets
# train_augmented.to_csv('train_augmented.csv')
# test_augmented.to_csv('test_augmented.csv')
# # train_augmented.head(), test_augmented.head()



# #########Q3#########
# # Note that the training set contains a ‘sales’ column while the test set does not. 
# # Use the training set to train a model of your choice and use that model to predict which values for sales should be in the test set. 
# # You should try training at least 2 models and compare their accuracy later.

# # Load the datasets
# oil_data = 'oil.csv'
# train_data = 'train.csv'
# test_data = 'test.csv'

# oil_df = pd.read_csv(oil_data, parse_dates=['date'])
# train_df = pd.read_csv(train_data, parse_dates=['date'])
# test_df = pd.read_csv(test_data, parse_dates=['date'])


# # Convert the 'date' column in oil data to datetime and set as index
# oil_df.set_index('date', inplace=True)

# # Perform linear interpolation to fill missing values in oil data
# oil_df_interpolated = oil_df.interpolate(method='linear')
# mean_oil_price = oil_df['dcoilwtico'].mean()
# oil_df_interpolated['dcoilwtico'].fillna(mean_oil_price, inplace=True) #fill na values with mean
# oil_df_interpolated.to_csv('oil3.csv')
# oil_df_inter_2 = pd.read_csv('oil3.csv', parse_dates=['date'])


# # Basic preprocessing for train and test datasets
# label_encoder = LabelEncoder()
# train_df['family_encoded'] = label_encoder.fit_transform(train_df['family'])
# test_df['family_encoded'] = label_encoder.transform(test_df['family'])

# # Extracting features from the 'date' column
# for df in [train_df, test_df]:
#     df['year'] = df['date'].dt.year
#     df['month'] = df['date'].dt.month
#     df['day'] = df['date'].dt.day
#     df['dayofweek'] = df['date'].dt.dayofweek

# # Merging with oil price data
# train_augmented = train_df.merge(oil_df_inter_2, left_on='date', right_index=True, how='left')
# test_augmented = test_df.merge(oil_df_inter_2, left_on='date', right_index=True, how='left')

# # Handling missing values with mean imputation .mean()
# train_augmented.fillna(train_augmented.mean(), inplace=True)
# test_augmented.fillna(test_augmented.mean(), inplace=True)

# # Preparing the data for training
# X_train = train_augmented.drop(['id', 'date', 'family', 'store_nbr', 'sales'], axis=1)
# y_train = train_augmented['sales']
# X_test = test_augmented.drop(['id', 'date', 'family', 'store_nbr'], axis=1)

# # Training Linear Regression and Random Forest models
# lr_model = LinearRegression()
# rf_model = RandomForestRegressor(random_state=42)
# # Random Forest Reg is a meta estimator that fits a number of classifying decision trees 
# #on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

# lr_model.fit(X_train, y_train)
# rf_model.fit(X_train, y_train)

# # Making predictions
# lr_predictions = lr_model.predict(X_test)
# rf_predictions = rf_model.predict(X_test)

# # If you need to evaluate model performance, you can split the training set and evaluate using RMSE:
# # X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
# # lr_val_predictions = lr_model.predict(X_val_split)
# # rf_val_predictions = rf_model.predict(X_val_split)
# # lr_rmse = mean_squared_error(y_val_split, lr_val_predictions, squared=False)
# # rf_rmse = mean_squared_error(y_val_split, rf_val_predictions, squared=False)

# # Save the predictions to a CSV file
# test_df['lr_sales_prediction'] = lr_predictions
# test_df['rf_sales_prediction'] = rf_predictions
# test_df.to_csv('predictions.csv', index=False)
