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
