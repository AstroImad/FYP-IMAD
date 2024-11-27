import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# List of files
files = ['16.data', '17.data']

# Load each file and combine them into one DataFrame
df_list = [pd.read_csv(file, sep='\s+') for file in files]
df = pd.concat(df_list, ignore_index=True)

# Prepare data for linear regression
x = df[['star_age']]
y = df[['log_L']]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Calculate and print the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R2 Score:', r2)

# Plotting the results
#plt.xscale('log')
plt.yscale('log')
plt.scatter(x_test, y_test, color='blue', label='Actual Data')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel(r'Star Age (Year)')
plt.ylabel(r'L /(L$_\odot$)')
plt.title('Linear Regression: Luminosity over Star Age')
plt.legend()
plt.show()
