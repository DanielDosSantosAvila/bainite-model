# This example shows how to fit experimental data to the model and extract the values of activation energies and
# carbon content

import pandas as pd
from bainitemodel import Composition, Bainite, TransformationKinetics, fit_single_curve

# Import experimental data
exp_data_excel = pd.read_excel('Example2_data.xlsx')
time_data_exp = exp_data_excel['Time (s)'].to_numpy()
bainite_fraction_exp = exp_data_excel['Bainite fraction'].to_numpy()
kinetics = TransformationKinetics(experiment_time_datapoints=time_data_exp,
                                  experiment_fraction_datapoints=bainite_fraction_exp)

# Define composition-related parameters
comp_params = Composition(t0prime_celsius=450, c1=2500, c2=7500, c=0.3, si=2.0, mn=4.3, cr=0.3)

# Define treatment-related parameters
bainite_params = Bainite(chemical_composition=comp_params, temp_celsius=400, grain_size=10e-6,
                         kinetics=kinetics)

# Fit the model to the experimental curve
# Using the 'differential_evolution' method for the minimizer results in a slower code, but it yields better results
fit = fit_single_curve(bainite_params=bainite_params, minimizer_options={'method': 'differential_evolution'})

# Plot the result
bainite_params.plot()

# Print the fitted parameters
bainite_params.print_parameters()
