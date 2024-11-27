import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# List of files
files = ['16.data', '17.data']

# Load each file and combine them into one DataFrame
df_list = [pd.read_csv(file, sep='\s+') for file in files]
df = pd.concat(df_list, ignore_index=True)

# Extract data
x = df['star_age'].values
y = df['log_L'].values

# Sort the data by x (star_age) to ensure a smooth interpolation
sorted_indices = np.argsort(x)
x = x[sorted_indices]
y = y[sorted_indices]

# Create the interpolation function
interpolation_function = interp1d(x, y, kind='linear', fill_value="extrapolate")

# Generate a smooth curve for interpolation
x_smooth = np.linspace(x.min(), x.max(), 500)  # 500 points for a smooth curve
y_smooth = interpolation_function(x_smooth)

# Plotting the original data
plt.scatter(df_list[0]['star_age'], df_list[0]['log_L'], color='blue', label='Data from 16.data', alpha=0.6)
plt.scatter(df_list[1]['star_age'], df_list[1]['log_L'], color='green', label='Data from 17.data', alpha=0.6)

# Plotting the interpolation curve
plt.plot(x_smooth, y_smooth, color='red', label='Interpolation Curve', linewidth=2)

# Customize the plot
plt.yscale('log')  # Log scale for y-axis
plt.xlabel(r'Star Age (Year)')
plt.ylabel(r'L /(L$_\odot$)')
plt.title('Interpolation of Combined Luminosity Data over Star Age')
plt.legend()
plt.show()
