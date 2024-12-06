import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels as sm
sns.set()

# Load the data from cars.csv
cardata = pd.read_csv('cars.csv')

# Initial data exploration
cardata.describe(include='all')

# Drop the 'Model' column
data = cardata.drop(['Model'], axis=1)
data.describe(include='all')

# Drop rows with missing values
data_no_rv = data.dropna(axis=0)
data_no_rv.describe(include='all')

# Visualize Price distribution (after missing values are dropped)
sns.displot(data_no_rv['Price'])
plt.show()

# Deal with outliers in 'Price'
q_price = data_no_rv['Price'].quantile(0.99)
data_price_in = data_no_rv[data_no_rv['Price'] < q_price]
data_price_in.describe(include='all')

# Visualize 'Price' after outlier removal
sns.displot(data_price_in['Price'])
plt.show()

# Deal with outliers in 'Mileage'
q_mileage = data_price_in['Mileage'].quantile(0.99)
data_mileage_in = data_price_in[data_price_in['Mileage'] < q_mileage]
data_mileage_in.describe(include='all')

# Visualize 'Mileage' after outlier removal
sns.displot(data_mileage_in['Mileage'])
plt.show()

# Deal with outliers in 'EngineV'
q_enginev = data_mileage_in['EngineV'].quantile(0.99)
data_enginev_in = data_mileage_in[data_mileage_in['EngineV'] < q_enginev]
data_enginev_in.describe(include='all')

# Visualize 'EngineV' after outlier removal
sns.displot(data_enginev_in['EngineV'])
plt.show()

# Deal with outliers in 'Year'
q_year = data_enginev_in['Year'].quantile(0.99)
data_year_in = data_enginev_in[data_enginev_in['Year'] < q_year]
data_cleaned = data_year_in.reset_index(drop=True)
data_cleaned.describe(include='all')

# Scatter plots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')

# Log transform the price
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price

# Define the target and inputs
targets = data_cleaned['log_price']
inputs = data_cleaned.drop(['log_price', 'Price'], axis=1)

# One-hot encode categorical variables
inputs = pd.get_dummies(inputs, drop_first=True)

# Ensure only numerical columns are scaled
numerical_columns = inputs.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
inputs[numerical_columns] = scaler.fit_transform(inputs[numerical_columns])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=365)
