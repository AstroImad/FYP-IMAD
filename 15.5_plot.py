import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Load the Excel file
data_file = '/Users/user/Documents/FYP/Coding/0p0/0p0/lnu_age_highmass_spline.xlsx'
df = pd.read_excel(data_file, sheet_name='Sheet1')

# Ensure Age12 (Myr) column is present
if 'Age12 (Myr)' not in df.columns:
    df['Age12 (Myr)'] = (df['Age11 (Myr)'] + df['Age13 (Myr)']) / 2

# Reorder the age columns to maintain proper sequence
age_columns = [col for col in sorted(df.columns) if "Age" in col]
mass_columns = [col for col in sorted(df.columns) if "Lnu_M" in col]

# Mass values corresponding to the columns
mass_values = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16])  # Mass labels

target_mass = 15  # Mass to interpolate for
ages_interpolated = []

# Perform interpolation for each row
for i in range(len(df)):
    corresponding_age_values = df.loc[i, age_columns].values.astype(float)
    
    # Interpolate age for mass 15.5 within the range of 15 and 16
    age_interp = interp1d([14, 16], [df.loc[i, 'Age14 (Myr)'], df.loc[i, 'Age16 (Myr)']], kind='linear')
    interpolated_age = age_interp(target_mass)
    ages_interpolated.append(interpolated_age)

# Add the interpolated Age column to the DataFrame
df['Interpolated Age (Myr)'] = ages_interpolated

# Extract relevant data for plotting
ages_10 = df['Age10 (Myr)'].astype(float)
lnu_values_10 = df['Lnu_M10 (erg/s)'].astype(float)

ages_11 = df['Age11 (Myr)'].astype(float)
lnu_values_11 = df['Lnu_M11 (erg/s)'].astype(float)

ages_12 = df['Age12 (Myr)'].astype(float)
lnu_values_12 = df['Lnu_M12 (erg/s)'].astype(float)

ages_13 = df['Age13 (Myr)'].astype(float)
lnu_values_13 = df['Lnu_M13 (erg/s)'].astype(float)

ages_14 = df['Age14 (Myr)'].astype(float)
lnu_values_14 = df['Lnu_M14 (erg/s)'].astype(float)

ages_15 = df['Age15 (Myr)'].astype(float)
lnu_values_15 = df['Lnu_M15 (erg/s)'].astype(float)

ages_16 = df['Age16 (Myr)'].astype(float)
lnu_values_16 = df['Lnu_M16 (erg/s)'].astype(float)

# Interpolate Lnu for mass 15.5 within the range of 15 and 16 solar masses
lnu_interp_14_15 = interp1d([14, 16], [lnu_values_14, lnu_values_16], axis=0, kind='linear')
lnu_values_14_5 = lnu_interp_14_15(target_mass)

#error
# e = target_mass - lnu_values_15
# plt.plot(10**ages1[-2], e)
# plt.ylabel("Luminosity offset",fontsize=16)
# plt.xlabel("Star age [yr]",fontsize=16)
# #plt.ylim(-0.1,0.1)
# plt.xscale('log')
# emax=max(e)
# emin=min(e)
# x=np.ones(len(predicted_ages[0]))
# plt.plot(10**ages1[-2], x*emax, color='blue', linestyle='--')
# plt.plot(10**ages1[-2], x*emin, color='blue', linestyle='--')
# plt.plot(10**ages1[-2], x*0, color='red', linestyle='--')
# plt.title(r'15 M$\odot$',fontsize=16)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# print(emax,emin)
# print(error)


# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(ages_10, lnu_values_10, marker='o',linestyle='-', label='10 Solar Mass')
plt.plot(ages_11, lnu_values_11, marker='o',linestyle='-', label='11 Solar Mass')
plt.plot(ages_12, lnu_values_12, marker='o',linestyle='-', label='12 Solar Mass')
plt.plot(ages_13, lnu_values_13, marker='o',linestyle='-', label='13 Solar Mass')
plt.plot(ages_14, lnu_values_14, marker='o', linestyle='-', label='14 Solar Mass')
plt.plot(ages_15, lnu_values_15, marker='o', linestyle='-', label='15 Solar Mass')
plt.plot(ages_16, lnu_values_16, marker='o', linestyle='-', label='16 Solar Mass')
plt.plot(df['Interpolated Age (Myr)'], lnu_values_15, marker='o', linestyle='--', label='15 Solar Mass')

# Labeling the axes and the plot
plt.title('Interpolated Luminosity vs Age for Different Masses', fontsize=16)
plt.xlabel('Age (Myr)', fontsize=14)
plt.ylabel('L$_{\\nu}$ (erg/s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
#plt.savefig('15.5 solar mass interpolated.png')
# Show the plot
plt.show()
