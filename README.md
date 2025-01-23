# FYP

FYP project

In this repo, there are several files describing the machine learning techniques in predicting the neutrino luminosity for the specific mass. The input data from this project is from https://zenodo.org/records/8327401 published by Farag et al (2023). For this project we take only one metallicity Z=0.02 from the folder 0p0.

The repo contains

1. interpolation_highmass_original.py #interpolation of neutrino luminosity vs age using two techniques - linear and B-spline for high mass stars
2. interpolation_lowmass_original.py #interpolation of neutrino luminosity vs age using two techniques - linear and B-spline for low mass stars
3. linearvspline_isocrohones.py #the data from (1) or (2) is rearrange for neutrino luminosity vs mass and we plot the data for each age
4. 15.5_plot.py # this is we extracted the data from step (1) and (2) and predict the neutrino luminosity for the specific mass
5. 15.5_plot_flux.py #this is the extended version of (4) where we add the flux calculation in the interpolated/estimated neutrino luminosity for specific mass (in this case 15.5 solar mass)
