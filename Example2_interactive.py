"""
When running this code an interactive Matplotlib window will open.
You can change the value of the sliders, and the plots should be automatically updated.
The title of the figure shows the weighted residuals.
Whenever you're happy with the match between experiments and the model, press the "Fit" button and the parameters
will start being fitted using lmfit, an Excel file with the simulated curve will be created, and a txt file with the
optimized parameters will be created.
"""
from bainitemodel import (Composition, TransformationKinetics, Bainite, bainite_model)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
import numpy as np
from lmfit import Parameters, minimize, fit_report


def main():

    # Import experimental data
    exp_data_excel = pd.read_excel('Example2_data.xlsx')
    time_data_exp = exp_data_excel['Time (s)'].to_numpy()
    bainite_fraction_exp = exp_data_excel['Bainite fraction'].to_numpy()
    kinetics = TransformationKinetics(experiment_time_datapoints=time_data_exp,
                                      experiment_fraction_datapoints=bainite_fraction_exp)
    # Define composition-related parameters
    comp_params = Composition(t0prime_celsius=450, c1=2500, c2=7500, c=0.3, si=2.0, mn=4.3, cr=0.3)

    # Define treatment-related parameters
    # If the sub-unit thickness is not specified, it will be calculated using van Bohemen's model (2018)
    bainite_params = Bainite(chemical_composition=comp_params, temp_celsius=400, grain_size=10e-6,
                             kinetics=kinetics)

    # Define initial parameters for plotting and for sliders
    q_gb_init = 234e03
    q_a_init = 221e03
    xb_init = 0.01029

    # Create plot and sliders
    f = bainite_fraction_exp
    t = time_data_exp
    dfdt = np.gradient(f, t)
    max_step_simulation = np.max(time_data_exp)/100
    f_simu = np.sum(bainite_model(bainite_params, act_energy_gb=q_gb_init, act_energy_autocatalysis=q_a_init,
                                  xb=xb_init, solver_options={'max_step': max_step_simulation}).y, axis=0)
    dfdt_simu = np.gradient(f_simu, t)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[0.7, 0.3])
    gs0 = gs[0].subgridspec(ncols=2, nrows=1)
    gs1 = gs[1].subgridspec(ncols=1, nrows=4)
    ax_f = fig.add_subplot(gs0[0])
    ax_dfdt = fig.add_subplot(gs0[1])
    line_exp, = ax_f.plot(t, f)
    line_simu, = ax_f.plot(t, f_simu)
    line1_exp, = ax_dfdt.plot(f, dfdt, 'o')
    line1_simu, = ax_dfdt.plot(f_simu, dfdt_simu)
    ax_f.set_xlabel('Time (s)')
    ax_f.set_ylabel('Bainite fraction')
    ax_dfdt.set_xlabel('Bainite fraction')
    ax_dfdt.set_ylabel('df/dt (1/s)')

    q_gb_axis = fig.add_subplot(gs1[0])
    q_gb_slider = Slider(
        ax=q_gb_axis,
        label='Q_GB',
        valmin=q_gb_init*0.8,
        valmax=q_gb_init*1.2,
        valinit=q_gb_init,
    )

    q_a_axis = fig.add_subplot(gs1[1])
    q_a_slider = Slider(
        ax=q_a_axis,
        label='Q_A',
        valmin=q_a_init*0.8,
        valmax=q_a_init*1.2,
        valinit=q_a_init,
    )

    xb_axis = fig.add_subplot(gs1[2])
    xb_slider = Slider(
        ax=xb_axis,
        label='Xb',
        valmin=0,
        valmax=comp_params.c_at_fr,
        valinit=xb_init,
    )

    def sum_residuals(modeled_fraction, experimental_fraction):
        return np.sum((modeled_fraction - experimental_fraction)**2/len(modeled_fraction))

    title = fig.suptitle(f'Residuals = {sum_residuals(f_simu, bainite_fraction_exp):.2e}')

    def update(_):
        f_simu_slider = np.sum(
            bainite_model(
                bainite_params, act_energy_gb=q_gb_slider.val, act_energy_autocatalysis=q_a_slider.val,
                xb=xb_slider.val, solver_options={'max_step': max_step_simulation}
            ).y,
            axis=0
        )
        dfdt_simu_slider = np.gradient(f_simu_slider, t)
        line_simu.set_ydata(f_simu_slider)
        line1_simu.set_xdata(f_simu_slider)
        line1_simu.set_ydata(dfdt_simu_slider)
        title.set_text(f'Residuals = {sum_residuals(f_simu_slider, bainite_fraction_exp):.2e}')
        fig.canvas.draw_idle()

    q_gb_slider.on_changed(update)
    q_a_slider.on_changed(update)
    xb_slider.on_changed(update)

    # Create fitting button and fitting parameters
    fitting_params = Parameters()

    def iter_cb(params, iter_num, resid, *args, **kwargs):
        print(iter_num, "\n", params, "\n", np.sum(resid))

    def residuals(params):
        q_gb = params['q_gb'].value
        q_a = params['q_a'].value
        xb = params['xb'].value
        simu = bainite_model(bainite_params, act_energy_gb=q_gb, act_energy_autocatalysis=q_a, xb=xb,
                             solver_options={'max_step': max_step_simulation})
        residual = bainite_params.kinetics.experiment_fraction_datapoints - np.sum(simu.y, axis=0)
        return residual

    def fit_curve(_):
        fitting_params.add(
            'q_gb',
            value=q_gb_slider.val,  # Initial guess for the optimization algorithm
            min=q_gb_slider.val - 10e3,  # Lower limit for the optimization algorithm
            max=q_gb_slider.val + 10e3,  # Upper limit for the optimization algorithm
            vary=True  # If set to False, the initial value will be fixed
        )

        fitting_params.add(
            'q_a',
            value=q_a_slider.val,  # Initial guess for the optimization algorithm
            min=q_a_slider.val - 10e3,  # Lower limit for the optimization algorithm
            max=q_a_slider.val + 10e3,  # Upper limit for the optimization algorithm
            vary=True  # If set to False, the initial value will be fixed
        )

        fitting_params.add(
            'xb',
            value=xb_slider.val,  # Initial guess for the optimization algorithm
            min=0,  # Lower limit for the optimization algorithm
            max=bainite_params.chemical_composition.c_at_fr,  # Upper limit for the optimization algorithm
            vary=True  # If set to False, the initial value will be fixed
        )

        fitting_minimizer = minimize(residuals, fitting_params, method='differential_evolution', iter_cb=iter_cb)
        print(fit_report(fitting_minimizer))
        bainite_params.act_energy_gb = fitting_minimizer.params['q_gb'].value
        bainite_params.act_energy_autocatalysis = fitting_minimizer.params['q_a'].value
        bainite_params.xb = fitting_minimizer.params['xb'].value
        opt_simu = bainite_model(bainite_params, solver_options={'max_step': max_step_simulation})
        df = pd.DataFrame({
            'time': opt_simu.t,
            'experimental_fraction': bainite_params.kinetics.experiment_fraction_datapoints,
            'simulated_fraction': np.sum(opt_simu.y, axis=0),
            'simulated f_gb': opt_simu.y[0],
            'simulated f_a': opt_simu.y[1]
        })
    # Choose name and path to export fitting results. If a file with the same names already exist, they will be
    # overwritten.
        df.to_excel('Example2_output.xlsx')
        with open('Example2_output.txt', 'w') as fr:
            fr.write(fit_report(fitting_minimizer))

    ax_button = fig.add_subplot(gs1[-1])
    button = Button(ax_button, 'Fit')
    button.on_clicked(fit_curve)

    plt.show()


if __name__ == '__main__':
    main()
