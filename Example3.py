# This example shows how to fit experimental data form multiple curves and extract the values of activation energies
# and carbon content

import pandas as pd
from bainitemodel import Composition, Bainite, TransformationKinetics, fit_multiple_curves

# Import experimental data for the steel with 10 µm prior austenite grain size
exp_data_excel_d10 = pd.read_excel('Example3_data_d10.xlsx')
time_data_exp_d10 = exp_data_excel_d10['Time (s)'].to_numpy()
bainite_fraction_exp_d10 = exp_data_excel_d10['Bainite fraction'].to_numpy()
kinetics_d10 = TransformationKinetics(experiment_time_datapoints=time_data_exp_d10,
                                      experiment_fraction_datapoints=bainite_fraction_exp_d10)

# Import experimental data for the steel with 15 µm prior austenite grain size
exp_data_excel_d15 = pd.read_excel('Example3_data_d15.xlsx')
time_data_exp_d15 = exp_data_excel_d15['Time (s)'].to_numpy()
bainite_fraction_exp_d15 = exp_data_excel_d15['Bainite fraction'].to_numpy()
kinetics_d15 = TransformationKinetics(experiment_time_datapoints=time_data_exp_d15,
                                      experiment_fraction_datapoints=bainite_fraction_exp_d15)

# Define composition-related parameters
# If the sub-unit thickness is not specified, it will be calculated using van Bohemen's model (2018)
comp_params = Composition(t0prime_celsius=450, c1=2500, c2=7500, c=0.3, si=2.0, mn=4.3, cr=0.3)

# Define treatment-related parameters
bainite_params_d10 = Bainite(chemical_composition=comp_params, temp_celsius=400, grain_size=10e-6,
                             kinetics=kinetics_d10)
bainite_params_d15 = Bainite(chemical_composition=comp_params, temp_celsius=400, grain_size=15e-6,
                             kinetics=kinetics_d15)

# Fit the model to the experimental curve
# Using the 'differential_evolution' method for the minimizer results in a slower code, but it yields better results
fit = fit_multiple_curves(bainite_params=[bainite_params_d10, bainite_params_d15],
                          minimizer_options={'method': 'differential_evolution'}
                          )

# Plot the result
bainite_params_d10.plot()
bainite_params_d15.plot()

# Print the fitted parameters
bainite_params_d10.print_parameters()
bainite_params_d15.print_parameters()
