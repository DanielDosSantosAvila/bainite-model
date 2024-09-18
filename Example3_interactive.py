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
    exp_data_excel_d10 = pd.read_excel('Example3_data_d10.xlsx')
    time_data_exp_d10 = exp_data_excel_d10['Time (s)'].to_numpy()
    bainite_fraction_exp_d10 = exp_data_excel_d10['Bainite fraction'].to_numpy()
    kinetics_d10 = TransformationKinetics(experiment_time_datapoints=time_data_exp_d10,
                                          experiment_fraction_datapoints=bainite_fraction_exp_d10)

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

    bainite_params_d15 = Bainite(chemical_composition=comp_params, temp_celsius=400, grain_size=10e-6,
                                 kinetics=kinetics_d15)

    # Define initial parameters for plotting and for sliders
    q_gb_init = 234e3
    q_a_init = 221e3
    xb_init = 0.01029

    # Create plot and sliders
    f_d10 = bainite_fraction_exp_d10
    f_d15 = bainite_fraction_exp_d15
    t_d10 = time_data_exp_d10
    t_d15 = time_data_exp_d15
    max_step_simulation = np.max(time_data_exp_d10)/100
    f_simu_d10 = np.sum(bainite_model(bainite_params_d10, act_energy_gb=q_gb_init, act_energy_autocatalysis=q_a_init,
                                      xb=xb_init, solver_options={'max_step': max_step_simulation}).y, axis=0)
    f_simu_d15 = np.sum(bainite_model(bainite_params_d15, act_energy_gb=q_gb_init, act_energy_autocatalysis=q_a_init,
                                      xb=xb_init, solver_options={'max_step': max_step_simulation}).y, axis=0)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[0.7, 0.3])
    gs0 = gs[0].subgridspec(ncols=2, nrows=1)
    gs1 = gs[1].subgridspec(ncols=1, nrows=5)
    ax_f_d10 = fig.add_subplot(gs0[0])
    ax_f_d15 = fig.add_subplot(gs0[1])
    ax_f_d10.plot(t_d10, f_d10)
    line_simu_d10, = ax_f_d10.plot(t_d10, f_simu_d10)
    ax_f_d15.plot(t_d15, f_d15)
    line_simu_d15, = ax_f_d15.plot(t_d15, f_simu_d15)
    ax_f_d10.set_xlabel('Time (s)')
    ax_f_d10.set_ylabel('Bainite fraction')
    ax_f_d15.set_xlabel('Time (s)')
    ax_f_d15.set_ylabel('Bainite fraction')

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

    xb_axis_d10 = fig.add_subplot(gs1[2])
    xb_slider_d10 = Slider(
        ax=xb_axis_d10,
        label='Xb_d10',
        valmin=0,
        valmax=comp_params.c_at_fr,
        valinit=xb_init,
    )

    xb_axis_d15 = fig.add_subplot(gs1[3])
    xb_slider_d15 = Slider(
        ax=xb_axis_d15,
        label='Xb_d15',
        valmin=0,
        valmax=comp_params.c_at_fr,
        valinit=xb_init,
    )

    def sum_residuals(modeled_fraction, experimental_fraction):
        return np.sum((modeled_fraction - experimental_fraction)**2/len(modeled_fraction))

    initial_sum_residuals = (sum_residuals(f_simu_d10, bainite_fraction_exp_d10)
                             + sum_residuals(f_simu_d15, bainite_fraction_exp_d15))

    title = fig.suptitle(f'Residuals = {initial_sum_residuals:.2e}')

    def update(_):
        f_simu_slider_d10 = np.sum(
            bainite_model(
                bainite_params_d10, act_energy_gb=q_gb_slider.val, act_energy_autocatalysis=q_a_slider.val,
                xb=xb_slider_d10.val, solver_options={'max_step': max_step_simulation}
            ).y,
            axis=0
        )
        line_simu_d10.set_ydata(f_simu_slider_d10)
        f_simu_slider_d15 = np.sum(
            bainite_model(
                bainite_params_d15, act_energy_gb=q_gb_slider.val, act_energy_autocatalysis=q_a_slider.val,
                xb=xb_slider_d15.val, solver_options={'max_step': max_step_simulation}
            ).y,
            axis=0
        )
        line_simu_d15.set_ydata(f_simu_slider_d15)
        slider_sum_residuals = (sum_residuals(f_simu_slider_d10, bainite_fraction_exp_d10)
                                + sum_residuals(f_simu_slider_d15, bainite_fraction_exp_d15))
        title.set_text(f'Residuals = {slider_sum_residuals:.2e}')
        fig.canvas.draw_idle()

    q_gb_slider.on_changed(update)
    q_a_slider.on_changed(update)
    xb_slider_d10.on_changed(update)
    xb_slider_d15.on_changed(update)

    # Create fitting button and fitting parameters
    fitting_params = Parameters()

    def iter_cb(params, iter_num, resid, *args, **kwargs):
        print(iter_num, "\n", params, "\n", np.sum(resid))

    def residuals(params):
        q_gb = params['q_gb'].value
        q_a = params['q_a'].value
        xb_d10 = params['xb_d10'].value
        xb_d15 = params['xb_d15'].value
        simu_d10 = bainite_model(bainite_params_d10, act_energy_gb=q_gb, act_energy_autocatalysis=q_a, xb=xb_d10,
                                 solver_options={'max_step': max_step_simulation})
        simu_d15 = bainite_model(bainite_params_d15, act_energy_gb=q_gb, act_energy_autocatalysis=q_a, xb=xb_d15,
                                 solver_options={'max_step': max_step_simulation})
        residual_d10 = bainite_params_d10.kinetics.experiment_fraction_datapoints - np.sum(simu_d10.y, axis=0)
        residual_d15 = bainite_params_d15.kinetics.experiment_fraction_datapoints - np.sum(simu_d15.y, axis=0)
        return np.concatenate((residual_d10/np.sqrt(len(residual_d10)),
                               residual_d15/np.sqrt(len(residual_d15))))

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
            'xb_d10',
            value=xb_slider_d10.val,  # Initial guess for the optimization algorithm
            min=0,  # Lower limit for the optimization algorithm
            max=bainite_params_d10.chemical_composition.c_at_fr,  # Upper limit for the optimization algorithm
            vary=True  # If set to False, the initial value will be fixed
        )

        fitting_params.add(
            'xb_d15',
            value=xb_slider_d15.val,  # Initial guess for the optimization algorithm
            min=0,  # Lower limit for the optimization algorithm
            max=bainite_params_d15.chemical_composition.c_at_fr,  # Upper limit for the optimization algorithm
            vary=True  # If set to False, the initial value will be fixed
        )

        fitting_minimizer = minimize(residuals, fitting_params, method='differential_evolution', iter_cb=iter_cb)
        print(fit_report(fitting_minimizer))
        bainite_params_d10.act_energy_gb = fitting_minimizer.params['q_gb'].value
        bainite_params_d10.act_energy_autocatalysis = fitting_minimizer.params['q_a'].value
        bainite_params_d15.act_energy_gb = fitting_minimizer.params['q_gb'].value
        bainite_params_d15.act_energy_autocatalysis = fitting_minimizer.params['q_a'].value
        bainite_params_d10.xb = fitting_minimizer.params['xb_d10'].value
        bainite_params_d15.xb = fitting_minimizer.params['xb_d15'].value
        opt_simu_d10 = bainite_model(bainite_params_d10, solver_options={'max_step': max_step_simulation})
        opt_simu_d15 = bainite_model(bainite_params_d15, solver_options={'max_step': max_step_simulation})
        df_d10 = pd.DataFrame({
            'time_d10': opt_simu_d10.t,
            'experimental_fraction_d10': bainite_params_d10.kinetics.experiment_fraction_datapoints,
            'simulated_fraction_d10': np.sum(opt_simu_d10.y, axis=0),
            'simulated f_gb_d10': opt_simu_d10.y[0],
            'simulated f_a_d10': opt_simu_d10.y[1]
        })
        df_d15 = pd.DataFrame({
            'time_d15': opt_simu_d15.t,
            'experimental_fraction_d15': bainite_params_d15.kinetics.experiment_fraction_datapoints,
            'simulated_fraction_d15': np.sum(opt_simu_d15.y, axis=0),
            'simulated f_gb_d15': opt_simu_d15.y[0],
            'simulated f_a_d15': opt_simu_d15.y[1]
        })
    # Choose name and path to export fitting results. If a file with the same names already exist, they will be
    # overwritten.
        df_d10.to_excel('Example3_output_d10.xlsx')
        df_d15.to_excel('Example3_output_d15.xlsx')
        with open('Example3_output.txt', 'w') as fr:
            fr.write(fit_report(fitting_minimizer))

    ax_button = fig.add_subplot(gs1[-1])
    button = Button(ax_button, 'Fit')
    button.on_clicked(fit_curve)

    plt.show()


if __name__ == '__main__':
    main()
