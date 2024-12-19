import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# List of files
files = ['16.data', '17.data']

# Load each file into a DataFrame
df_list = [pd.read_csv(file, sep='\s+') for file in files]

# Mass values associated with each file
masses = [16, 17]

# Extract x and y from each DataFrame
x1, y1 = df_list[0]['star_age'].values, df_list[0]['log_L'].values
x2, y2 = df_list[1]['star_age'].values, df_list[1]['log_L'].values

# Create a common grid for x (star_age) spanning both datasets
common_x = np.linspace(min(x1.min(), x2.min()), max(x1.max(), x2.max()), 500)

# Interpolate y values for both datasets onto the common x grid
interp1 = interp1d(x1, y1, kind='linear', fill_value="extrapolate")
interp2 = interp1d(x2, y2, kind='linear', fill_value="extrapolate")

common_y1 = interp1(common_x)  # Interpolated values from 16.data
common_y2 = interp2(common_x)  # Interpolated values from 17.data

# Average the interpolated y values
combined_y = (common_y1 + common_y2) / 2

# Interpolated mass for the red line
interpolated_mass = (masses[0] + masses[1]) / 2
print(f"Interpolated Mass for the Red Line: {interpolated_mass:.1f}")

# Plotting the original data
plt.scatter(x1, y1, color='blue', label='Data from 16.data (Mass = 16)', alpha=0.6)
plt.scatter(x2, y2, color='green', label='Data from 17.data (Mass = 17)', alpha=0.6)

# Plotting the combined interpolation curve
plt.plot(common_x, combined_y, color='red', label=f'Interpolation Curve (Mass = {interpolated_mass:.1f})', linewidth=2)

# Customize the plot
plt.yscale('log')  # Log scale for y-axis
plt.xlabel(r'Star Age (Year)')
plt.ylabel(r'L /(L$_\odot$)')
plt.title('Interpolation of Combined Luminosity Data over Star Age')
plt.legend()
plt.show()
