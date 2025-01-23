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

# Interpolation for 14.5 solar mass
target_mass_14_5 = 14.5
ages_interpolated_14_5 = []

# Perform interpolation for each row (14.5 solar mass)
for i in range(len(df)):
    corresponding_age_values = df.loc[i, age_columns].values.astype(float)
    
    # Interpolate age for mass 14.5 within the range of 14 and 15
    age_interp_14_5 = interp1d([14, 15], [df.loc[i, 'Age14 (Myr)'], df.loc[i, 'Age15 (Myr)']], kind='linear')
    interpolated_age_14_5 = age_interp_14_5(target_mass_14_5)
    ages_interpolated_14_5.append(interpolated_age_14_5)

lnu_interp_14_15 = interp1d([14, 15], [df['Lnu_M14 (erg/s)'], df['Lnu_M15 (erg/s)']], axis=0, kind='linear')
lnu_values_14_5 = lnu_interp_14_15(target_mass_14_5)

# Interpolation for 15.5 solar mass
target_mass_15_5 = 15.5
ages_interpolated_15_5 = []

# Perform interpolation for each row (15.5 solar mass)
for i in range(len(df)):
    corresponding_age_values = df.loc[i, age_columns].values.astype(float)
    
    # Interpolate age for mass 15.5 within the range of 15 and 16
    age_interp_15_5 = interp1d([15, 16], [df.loc[i, 'Age15 (Myr)'], df.loc[i, 'Age16 (Myr)']], kind='linear')
    interpolated_age_15_5 = age_interp_15_5(target_mass_15_5)
    ages_interpolated_15_5.append(interpolated_age_15_5)

lnu_interp_15_16 = interp1d([15, 16], [df['Lnu_M15 (erg/s)'], df['Lnu_M16 (erg/s)']], axis=0, kind='linear')
lnu_values_15_5 = lnu_interp_15_16(target_mass_15_5)

# Calculate flux for 14.5 solar mass at a distance of 1040 light-years
distance_14_5_cm = 1040 * 9.461e17  # Convert light-years to cm
flux_values_14_5 = lnu_values_14_5 / (4 * np.pi * distance_14_5_cm**2)

# Calculate flux for 15.5 solar mass at a distance of 650 light-years
distance_15_5_cm = 650 * 9.461e17  # Convert light-years to cm
flux_values_15_5 = lnu_values_15_5 / (4 * np.pi * distance_15_5_cm**2)

# Extract relevant data for plotting
ages_8 = df['Age8 (Myr)'].astype(float)
lnu_values_8 = df['Lnu_M8 (erg/s)'].astype(float)

ages_9 = df['Age9 (Myr)'].astype(float)
lnu_values_9 = df['Lnu_M9 (erg/s)'].astype(float)

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



# Create the plot
fig, axs = plt.subplots(1, 2, figsize=(18, 8))

# Plot Lnu vs Age
axs[0].plot(ages_8, lnu_values_8, linestyle='-', label='8 $M_\odot$')
axs[0].plot(ages_9, lnu_values_9, linestyle='-', label='9 $M_\odot$')
axs[0].plot(ages_10, lnu_values_10, linestyle='-', label='10 $M_\odot$')
axs[0].plot(ages_11, lnu_values_11, linestyle='-', label='11 $M_\odot$')
axs[0].plot(ages_12, lnu_values_12, linestyle='-', label='12 $M_\odot$')
axs[0].plot(ages_13, lnu_values_13, linestyle='-', label='13 $M_\odot$')
axs[0].plot(ages_14, lnu_values_14, linestyle='-', label='14 $M_\odot$')
axs[0].plot(ages_15, lnu_values_15, linestyle='-', label='15 $M_\odot$')
axs[0].plot(ages_16, lnu_values_16, linestyle='-', label='16 $M_\odot$')
axs[0].plot(ages_interpolated_14_5, lnu_values_14_5, marker='o', linestyle='--', label='14.5 $M_\odot$ (interpolated)')
axs[0].plot(ages_interpolated_15_5, lnu_values_15_5, marker='o', linestyle='--', label='15.5 $M_\odot$ (interpolated)')
axs[0].set_title('Interpolated Luminosity ($L_\\nu$) vs Age', fontsize=16)
axs[0].set_xlabel('Age (Myr)', fontsize=14)
axs[0].set_ylabel('$L_\\nu$ (erg/s)', fontsize=14)
#axs[0].set_ylim(-5, 10)
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(fontsize=12)

# Plot Flux vs Age
axs[1].plot(ages_interpolated_14_5, flux_values_14_5, marker='o', linestyle='--', color='green', label='Flux for 14.5 $M_\odot$ at 1040 ly ($\\beta$ Doradus)')
axs[1].plot(ages_interpolated_15_5, flux_values_15_5, marker='o', linestyle='--', color='purple', label='Flux for 15.5 $M_\odot$ at 650 ly (Saiph)')
axs[1].set_title('Flux vs Age for 14.5 and 15.5 $M_\odot$', fontsize=16)
axs[1].set_xlabel('Age (Myr)', fontsize=14)
axs[1].set_ylabel('Flux, $\phi$ (erg/s/cm$^2$)', fontsize=14)
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(fontsize=12)

# Show the plots
plt.tight_layout()
plt.savefig('interpolation_various_mass.png')
plt.show()