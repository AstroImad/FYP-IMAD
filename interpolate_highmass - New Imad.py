import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splprep, splev

#------------plotting paramater------------
params = {'backend': 'tkagg',
          'axes.labelsize':  18,
#          'linewidth':   1,
          'legend.fontsize': 12,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'figure.figsize': (8,8),
          'text.usetex':True,
          'axes.titlesize': 18 ,
          'figure.dpi': 90}
plt.rcParams.update(params)

# File reading
file_prefix = '/Users/user/Documents/FYP/Coding/0p0/0p0/'
mass_range = range(8, 17)  # Masses from 8 to 30
dfs = [pd.read_csv(f'{file_prefix}{mass}.data', header=0, sep='\s+') for mass in mass_range]

# Extracting x and y data
x_data = [df['star_age'] / 1e7 for df in dfs]
y_data = [df['log_Lneu'] for df in dfs]

# Interpolation
x_inter = np.linspace(0, 4.0, 10001)
linear_interpolations = [interp1d(x, y, kind='linear', fill_value="extrapolate")(x_inter) for x, y in zip(x_data, y_data)]
splprep_interpolations = [splprep([x, y], s=0) for x, y in zip(x_data, y_data)]
spline_results = [splev(np.linspace(0, 1, 1000), tck) for tck, _ in splprep_interpolations]

# Tabulate results into DataFrames
data1 = {"Age (Myr)": x_inter}
for i, y_new in enumerate(linear_interpolations, start=1):
    data1[f"Lnu_M{mass_range[i-1]} (erg/s)"] = y_new

df_linear = pd.DataFrame(data1)
df_linear.to_excel("output_highmass_linear.xlsx")

# Every 100th row
df_linear.iloc[::100].T.to_excel("lnu_age_highmass_linear.xlsx")

data2 = {"Age (Myr)": spline_results[0][0]}
for i, (x_spline, y_spline) in enumerate(spline_results, start=1):
    data2[f"Lnu_M{mass_range[i-1]} (erg/s)"] = y_spline

df_spline = pd.DataFrame(data2)
df_spline.to_excel("output_highmass_spline.xlsx")
df_spline.iloc[::100].T.to_excel("lnu_age_highmass_spline.xlsx")

# Plotting
plt.figure(1)
plt.title('Linear Interpolation Fitting')
plt.xlabel('Star Age ($10^7$ yr)')
plt.ylabel('Neutrino Luminosity (L$_\\nu$/L$_\odot$)')
colors = plt.cm.jet(np.linspace(0, 1, len(mass_range)))

for i, mass in enumerate(mass_range):
    plt.plot(x_data[i], y_data[i], label=f'{mass}M$_\\odot$', color=colors[i])
    plt.scatter(x_inter, linear_interpolations[i], color='gray', label=f'Interp {mass}M$_\\odot$')

    # Alternate colors (gray and yellow)
    #colors = 'gray' if i % 2 == 0 else 'yellow'
    #plt.scatter(x_inter, linear_interpolations[i], c=colors[i], label='interp y[i]')

plt.legend(bbox_to_anchor=(1.02, 1.15), loc='upper left')#, fontsize=7)
plt.tight_layout()
plt.savefig('Linear Interpolation.png')

plt.figure(2)
plt.title('B-Spline Fitting')
plt.xlabel('Star Age ($10^7$ yr)')
plt.ylabel('Neutrino Luminosity (L$_\\nu$/L$_\odot$)')

for i, mass in enumerate(mass_range):
    x_spline, y_spline = spline_results[i]
    plt.plot(x_data[i], y_data[i], label=f'{mass}M$_\\odot$', color=colors[i])
    plt.scatter(x_spline, y_spline, color='gray', label=f'B-Spline {mass}M$_\\odot$')

    # Alternate colors (gray and yellow)
    #colors = 'gray' if i % 2 == 0 else 'yellow'
    #plt.scatter(x_spline, y_spline, c=colors[i], label='B-Spline')

plt.legend(bbox_to_anchor=(1.02, 1.15),loc='upper left')#, fontsize=7)
plt.tight_layout()
plt.savefig('B-Spline Fitting.png')
plt.show()
