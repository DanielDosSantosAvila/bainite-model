# This example shows how to simulate a curve of bainite formation given a set of parameters

from bainitemodel import Composition, Bainite, bainite_model
import numpy as np
import matplotlib.pyplot as plt

# Define composition-related parameters
comp_params = Composition(c=0.2, mn=1.5, cr=1)

# Define treatment-related parameters
bainite_params = Bainite(chemical_composition=comp_params, temp_celsius=300, grain_size=20e-6, act_energy_gb=210e3,
                         act_energy_autocatalysis=205e3, xb=comp_params.c_at_fr)

# Simulate curve
time = np.linspace(0, 3600, 3600)
simulated_curve = bainite_model(bainite_params, time=time)
f_gb = simulated_curve.y[0]
f_autocatalysis = simulated_curve.y[1]
f_total = f_gb + f_autocatalysis

# Plot curve
fig, ax = plt.subplots()
ax.plot(time, f_gb, ls='dotted', color='black', label='$f_{gb}$')
ax.plot(time, f_autocatalysis, ls='dashed', color='black', label='$f_{a}$')
ax.plot(time, f_total, ls='solid', color='black', label="$f$")
ax.set_xlim(0, 3600)
ax.set_ylim(0, 1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Bainite fraction')
ax.legend()
fig.show()
