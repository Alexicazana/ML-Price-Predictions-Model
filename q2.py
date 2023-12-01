import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


#########Q2#########
# Re-loading the train.csv and test.csv files
train_data = 'train.csv'
test_data = 'test.csv'

train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

# # Convert the 'date' column in train and test datasets to datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# # Merging the train and test datasets with the oil price data
train_augmented = train_df.merge(oil_df_interpolated, left_on='date', right_index=True, how='left')
test_augmented = test_df.merge(oil_df_interpolated, left_on='date', right_index=True, how='left')

# # Displaying the first few rows of the augmented datasets
train_augmented.to_csv('train_augmented.csv')
test_augmented.to_csv('test_augmented.csv')
# train_augmented.head(), test_augmented.head()
