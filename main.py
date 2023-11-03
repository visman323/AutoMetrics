import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

cardata = pd.read_csv('cars.csv')


cardata.describe(include='all')

data = cardata.drop(['Model'],axis=1)
data.describe(include='all')

data_no_rv = data.dropna(axis=0)
data_no_rv.describe(include='all')

#datasns = sns.load_dataset("data_no_rv")

sns.displot(data_no_rv['Price'])

plt.show()

#Deal with outliers

q = data_no_rv['Price'].quantile(0.99)
data_price_in = data_no_rv[data_no_rv['Price']<q]
data_price_in.describe(include='all')

sns.displot(data_no_rv['Price'])

plt.show()

#seaborn.countplot(x="Model", data=data_no_rv)
