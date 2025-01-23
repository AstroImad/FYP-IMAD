import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d, splprep, splev, CubicSpline
#from scipy.interpolate import splprep, splev
import pandas as pd


#------------plotting paramater------------
params = {'backend': 'MacOSX',
          'axes.labelsize':  16,
#          'linewidth':   1,
          'legend.fontsize': 14,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'figure.figsize': (8,8),
          'text.usetex':True,
          'figure.dpi': 90}
plt.rcParams.update(params)
#--

dflinear = pd.read_excel('lnu_age_highmass_linear.xlsx')
dfspline = pd.read_excel('lnu_age_highmass_spline.xlsx')

#linear 

# Correct mass extraction: Using the headers as mass values
mass_labels = dflinear.iloc[1:, 0]  # Extract mass labels (Lnu_M8, Lnu_M9, etc.)
masses = mass_labels.str.extract(r'(\d+)').astype(float)  # Extract numerical part

# Filter the data to include every 50th column (to reduce the number of ages plotted)
filtered_columns = [0] + list(range(1, dflinear.shape[1], 10))  # Keep the first column and every 50th column
filtered_data = dflinear.iloc[:, filtered_columns]

# Update ages for the filtered data
filtered_ages = filtered_data.iloc[0, 1:].astype(float)

# Figure 1: Luminosity vs. Mass for Different Ages
plt.figure(1, figsize=(12, 8))

# Loop through the filtered columns
for col_index in range(1, filtered_data.shape[1]):
    age_label = filtered_data.iloc[0, col_index]  # Age identifier
    luminosity = filtered_data.iloc[1:, col_index].replace([float('inf'), -float('inf')], None).astype(float)
    
    # Plot the luminosity vs. mass
    plt.plot(masses, luminosity, label=f'Age: {age_label} Myr')

# Plot formatting
plt.ylim(0,4)
plt.xlim(7.5,16)
plt.xlabel('Mass', fontsize=14)
plt.ylabel('Luminosity (erg/s)', fontsize=14)
plt.title('Luminosity vs Mass for Different Ages', fontsize=16)
plt.legend(loc='best', fontsize=10)
#plt.grid(True)
plt.savefig('neu_isocrone_linear.png')
plt.tight_layout()

#spline



plt.figure(2, figsize=(12, 8))
# First dataset (every 50th column)
filtered_columns = [0] + list(range(1, dfspline.shape[1], 1000))
filtered_data = dfspline.iloc[:, filtered_columns]
filtered_ages = filtered_data.iloc[0, 1:].astype(float)

# Second dataset (age columns: x1, x2, ...)
ages_spline = dfspline.filter(like='x').iloc[0, :].astype(float)  # Extract x columns (ages)
luminosities_spline = dfspline.filter(like='Lnu')  # Extract luminosity columns

# Inspect the raw values of `ages_spline` to check for discrepancies
dfspline.filter(like='x').head(), dfspline.filter(like='Lnu').head()

# Extract mass values from column names for spline data
masses_spline = luminosities_spline.columns.str.extract(r'(\d+)').astype(float).squeeze()


# Interpolate luminosity at mass 15.5 for each row in the spline dataset
mass_to_interpolate = 15.5
interpolated_luminosities = []

for index, row in dfspline.iterrows():
    luminosities = row.filter(like='Lnu').astype(float)  # Extract luminosity values
    interpolation = np.interp(mass_to_interpolate, masses_spline, luminosities)  # Interpolate luminosity
    interpolated_luminosities.append(interpolation)

# Create a DataFrame to tabulate the results
interpolated_table = pd.DataFrame({
    'Row': range(1, len(interpolated_luminosities) + 1),
    'Luminosity at 15.5 M': interpolated_luminosities
})

interpolated_table.to_excel('15_5_interpolated_data.xlsx', index=False)


# Loop through each row in the spline dataset
for index, row in dfspline.iterrows():
    luminosities = row.filter(like='Lnu').astype(float)  # Extract luminosity values

    # Plot mass vs. luminosity for the current row
    plt.plot(masses_spline, luminosities, '--', label=f'Age{index + 1}')

# Plot formatting for Figure 2

plt.axvline(x=15.5, color='b')
plt.xlabel('Mass', fontsize=14)
plt.ylabel('Luminosity (erg/s)', fontsize=14)
plt.title('Luminosity vs Mass for Spline Data', fontsize=16)
plt.legend(bbox_to_anchor=(1.00, 1.0), loc='upper left')
#plt.grid(True)
plt.savefig('neu_isocrone_spline.png')
plt.tight_layout()

# Show the corrected figure
plt.show()
