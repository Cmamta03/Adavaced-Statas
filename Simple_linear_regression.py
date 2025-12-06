import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
print("Dataset Shape:", dataset.shape)

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color='green')
plt.plot(x_train, regressor.predict(x_train), color='pink')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m = regressor.coef_

c = regressor.intercept_

y_12 = 9312 * 12 + 26780

y_20 = 9312 * 20 + 26780

bias = regressor.score(x_train, y_train)
bias

variance = regressor.score(x_test, y_test)
variance

import pickle

filename = 'linear_regression_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
    
print("Model has been pickled and saved as linear_regression_model.pkl")

print(f"Intercept: {regressor.intercept_}")
print(f"Coefficient: {regressor.coef_}")

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

dataset.mean()
dataset.std()
dataset.var()

from scipy.stats import variation
variation(dataset.values)

variation(dataset['Salary'])
dataset.corr()
dataset.skew()
dataset.sem()

import scipy.stats as stats
dataset.apply(stats.zscore)

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1 - (SSR/SST)
r_square

print(regressor)

coef = print(f"Coefficient: {regressor.coef_}")
intercept = print(f"Intercept: {regressor.intercept_}")

exp_12_future_pred = 9312 * 100 + 26780
exp_12_future_pred

dataset.mean()