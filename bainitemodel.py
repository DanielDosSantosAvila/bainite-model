import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import constants
from scipy.integrate import solve_ivp
from periodictable import elements
from warnings import warn
from lmfit import Parameters, minimize, minimizer

# Physical constants used in the model
k = constants.Boltzmann
r = constants.gas_constant
h = constants.Planck


class Composition:
    """
    Store the chemical composition (wt.%) of the steel and the composition-related parameters as attributes.

    Parameters
    ----------
    t0prime_celsius: float, optional
        T0' of the steel in Celsius degrees. Default is 700.
    c1: float, optional
        Constant used for calculating the change in Th as a function of carbon content of the austenite,
        in Kelvin/at.fraction. Default is 0.
    c2: float, optional
        Constant used for calculating the change in T0'as a function of carbon content of the austenite,
        in Kelvin/at.fraction. Default is 0.
    **comp: float, optional
        Content of given element in wt.%. Key is the element symbol and value is the wt.%.
        Default is 0 for every element, except Fe which is balanced.

    Attributes
    ----------
    c_at_fr: float
        Carbon content in atomic fraction. Calculated automatically from the comp_wt_pct in wt.%.

    """

    def __init__(self, *, t0prime_celsius=727, c1=0, c2=0, **comp: float):
        self.t0prime_celsius = t0prime_celsius
        self.c1 = c1
        self.c2 = c2
        els = [el.symbol.lower() for el in elements if el.symbol not in ['n', 'Fe']]
        comp_lowercase = {el.lower(): wt for el, wt in comp.items()}
        balance = 100
        for el in comp_lowercase.keys():
            if el not in els:
                raise SyntaxError(f'{el} is not a valid element.')
        for el in els:
            wt = comp_lowercase.get(el, 0)
            if not 0 <= wt <= 100:
                raise ValueError('Wt.% should be between 0 and 100.')
            self.__setattr__(el, wt)
            balance -= wt
        if 'fe' in comp_lowercase:
            self.fe = comp_lowercase['fe']
        else:
            comp_lowercase['fe'] = balance
            self.fe = balance
        if np.round(np.array([wt_pct for wt_pct in comp_lowercase.values()]).sum(), 0) != 100:
            raise ValueError('Elements should add up to 100%.')
        self.comp_dict = comp_lowercase
        self.c_at_fr = carbon_wt_pct_to_at_fr(self)
        self.k_gamma = k_gamma_van_bohemen(self)

    def __str__(self):
        printed_composition = 'Fe'
        for el, wt_pct in self.comp_dict.items():
            if el != 'fe' and wt_pct > 0:
                printed_composition += f"-{np.round(wt_pct, 2)}{el.capitalize()}"
        return printed_composition


def carbon_wt_pct_to_at_fr(comp_wt_pct: Composition) -> float:
    """
    Converts the chemical comp_wt_pct from weight percent to atomic fraction.

    Parameters
    ----------
    comp_wt_pct : dictionary
        Keys: element symbol.
        Values: element content in wt.%.

    Returns
    -------
    c_at_fr : float
        Atomic fraction of carbon.

    """
    comp_wt_pct = {el.capitalize(): wt_pct for el, wt_pct in comp_wt_pct.comp_dict.items()}

    if not comp_wt_pct.get('Fe'):
        balance = 100 - np.sum([wt for wt in comp_wt_pct.values()])
        comp_wt_pct['Fe'] = balance

    mol_per_gram_steel = {}

    for (el, wt_pc) in comp_wt_pct.items():
        mol = wt_pc / (elements.symbol(el).mass * 100)
        mol_per_gram_steel[el] = mol

    mol_sum = np.sum(np.fromiter(mol_per_gram_steel.values(), dtype=float))
    comp_at_fr = {}

    for el, mol_per_gram in mol_per_gram_steel.items():
        at_fr = mol_per_gram / mol_sum
        comp_at_fr[el] = at_fr

    c_at_fr = comp_at_fr['C']

    return c_at_fr


def k_gamma_van_bohemen(comp_wt_pct: Composition) -> float:
    """
    Calculate the K_gamma parameter using the equation from van Bohemen.

    Parameters
    ----------
    comp_wt_pct: Composition
        Instance of Composition containing the chemical composition of the steel.

    Returns
    -------
    k_gamma: float
        K_gamma in joules per kelvin per mol.

    References
    ----------
    van Bohemen, S. M. C. (2010). Modeling start curves of bainite formation.
    Metallurgical and Materials Transactions A, 41, 285-296.

    """
    c = getattr(comp_wt_pct, 'c', 0)
    cr = getattr(comp_wt_pct, 'cr', 0)
    mn = getattr(comp_wt_pct, 'mn', 0)
    mo = getattr(comp_wt_pct, 'mo', 0)
    ni = getattr(comp_wt_pct, 'ni', 0)
    si = getattr(comp_wt_pct, 'si', 0)

    k_gamma = 1e3 * ((170 - 89 * c - 10 * mn - 12 * si - 2 * cr - 1 * ni - 29 * mo) / 705)

    return k_gamma


def sub_unit_thickness_van_bohemen(comp_wt_pct: Composition, temp_celsius: float, grain_size: float) -> float:
    """
    Calculate the thickness of the bainite sub-unit using the model from van Bohemen.
    If the parameters are outside the validity range of the model, the thickness wil be set equal to 0.200 micrometers.

    Parameters
    ----------
    comp_wt_pct: Composition
        Instance of Composition containing the chemical composition of the steel.
    temp_celsius: float
        Temperature of bainite formation in Celsius degrees.
    grain_size: float
        Prior austenite grain size in meters.

    Returns
    -------
    sub_unit_thickness: float
        Thickness of the bainite sub-unit in meters.

    References
    ----------
    van Bohemen, S. M. C. (2018). Exploring the correlation between the austenite yield strength and the bainite lath
    thickness. Materials Science and Engineering: A, 731, 119-123.

    """
    al = getattr(comp_wt_pct, 'al', 0)
    c = getattr(comp_wt_pct, 'c', 0)
    co = getattr(comp_wt_pct, 'co', 0)
    cr = getattr(comp_wt_pct, 'cr', 0)
    mn = getattr(comp_wt_pct, 'mn', 0)
    mo = getattr(comp_wt_pct, 'mo', 0)
    nb = getattr(comp_wt_pct, 'nb', 0)
    ni = getattr(comp_wt_pct, 'ni', 0)
    si = getattr(comp_wt_pct, 'si', 0)
    ti = getattr(comp_wt_pct, 'ti', 0)
    v = getattr(comp_wt_pct, 'v', 0)
    w = getattr(comp_wt_pct, 'w', 0)

    ys_room_t = (87.8 + 254 * c + 15.1 * si + 2.5 * cr + 14.5 * mo + 18.5 * v + 4.5 * w + 40 * nb
                 + 5.4 * al + 26.2 * ti + 10.5 / np.sqrt(grain_size * 1000))
    ys_lim = (66.6 + 140 * c - 1.1 * mn + 1.8 * si + 2.6 * cr + 7.7 * mo - 0.8 * ni
              + 1.5 * al + 2.2 * co + 4.9 / np.sqrt(grain_size * 1000))
    ys = ys_room_t * (1 - 2.2 * (temp_celsius - 25) * 1e-3 + 4.2 * ((temp_celsius - 25) ** 2) * 1e-6
                      - 3.0 * ((temp_celsius - 25) ** 3) * 1e-9)

    sub_unit_thickness = 2e-6 / (ys - ys_lim)

    # Check if values are within the validity range
    delta_ys = (ys - ys_lim)
    if not 5 < delta_ys < 70:
        warn("Conditions are outside the validity range of van Bohemen's model. The bainite sub-unit thickness"
             "will be set equal to 0.200 micrometers.")
        sub_unit_thickness = 0.2e-6

    return sub_unit_thickness


@dataclass
class TransformationKinetics:
    """
    Store the data points of bainite transformation from experiment and/or simulation.

    Parameters
    ----------
    experiment_time_datapoints: numpy.ndarray, optional
        Time datapoints of the experiment, in seconds. Default is empty array.
    experiment_fraction_datapoints: numpy.ndarray, optional
        Bainite fraction datapoints of the experiment, in volume fraction. Default is empty array.
    simulation_time_datapoints: numpy.ndarray, optional
        Time datapoints of the simulation, in seconds. Default is empty array.
    simulation_fraction_datapoints: numpy.ndarray, optional
        Bainite fraction datapoints of the simulation, in volume fraction. Default is empty array.

    """
    experiment_time_datapoints: np.ndarray = np.array([])
    experiment_fraction_datapoints: np.ndarray = np.array([])
    simulation_time_datapoints: np.ndarray = np.array([])
    simulation_fraction_datapoints: np.ndarray = np.array([])


@dataclass
class Bainite:
    """
    Store all relevant parameters of a given bainitic transformation as attributes.

    Parameters
    ----------
    chemical_composition: Composition
        Composition-related parameters of the steel
    temp_celsius: float
        Temperature of bainite formation, in Celsius degrees.
    grain_size: float
        Prior austenite grain size, in meters.
    grain_shape: float, optional
        Shape factor of the prior austenite grains. Default is 3.35 (tetradecahedron).
    sub_unit_thickness: float, optional
        Thickness of the bainite sub-unit, in meters. If None, the thickness is calculated using the model from
        van Bohemen, 2018. Default is None.
    sub_unit_aspect_ratio: float, optional
        Aspect ratio of the bainite sub-unit, calculated as length divided by thickness. Default is 6.
    act_energy_gb: float, optional
        Activation energy for bainite nucleation at the austenite grain boundaries, in J/mol. Default is None.
    act_energy_autocatalysis: float, optional
        Activation energy for bainite nucleation by autocalysis, in J/mol. Default is None.
    xb: float, optional
        Carbon content trapped in the bainite, in atomic fraction. Default is None.
    n_s_gb: float, optional
        Density of potential bainite nucleation sites per unit area of austenite grain boundary, in sites per square
        meter. Default is 1e16.
    n_s_autocatalysis: float, optional
        Density of potential bainite nucleation sites per unit area of bainite/austenite interface, in sites per square
        meter. Default is 1e16.
    kinetics: TransformationKinetics, optional
        Experimental and/or simulated data of the bainite transformation kinetics. Default is instance of
        TransformationKinetics containing empty arrays.
    name: string, optional
        Name of the treatment.

    """
    chemical_composition: Composition
    temp_celsius: float
    grain_size: float
    grain_shape: float = 3.35
    sub_unit_thickness: float = None
    sub_unit_aspect_ratio: float = 6
    act_energy_gb: float = None
    act_energy_autocatalysis: float = None
    xb: float = None
    n_s_gb: float = 1e16
    n_s_autocatalysis: float = 1e16
    kinetics: TransformationKinetics = TransformationKinetics()
    name: str = None

    def __post_init__(self):
        if self.sub_unit_thickness is None:
            self.sub_unit_thickness = sub_unit_thickness_van_bohemen(self.chemical_composition,
                                                                     self.temp_celsius,
                                                                     self.grain_size)
        if self.name is None:
            self.name = f"{self.chemical_composition.__str__()}_{int(self.temp_celsius)}°C_d=" \
                        f"{np.round(self.grain_size*1e6, 0)}µm"
        return None

    @property
    def sub_unit_length(self):
        """
        float: Length of the bainite sub-unit in meters calculated using the thickness and aspect ratio of the sub-unit.
        If larger than the austenite grain size, it will be set equal to the grain size.

        """
        sub_unit_length = self.sub_unit_thickness * self.sub_unit_aspect_ratio
        if sub_unit_length > self.grain_size:
            warn(f'sub_unit_length will be set equal to the grain size ({self.grain_size}).')
            sub_unit_length = self.grain_size
        return sub_unit_length

    @property
    def sub_unit_volume(self):
        """
        float: Volume of the bainite sub-unit in cubic meters calculated in using the thickness and aspect ratio of
        the sub-unit.

        """
        return self.sub_unit_thickness * (self.sub_unit_length ** 2)

    @property
    def f_gb_max(self):
        """
        float: Maximum fraction of bainite that can be formed by grain boundary nucleation calculated using the
        grain size and shape and the length of the sub-unit.
        If larger than one, it will be set equal to one.

        """
        f_gb_max = self.grain_shape * self.sub_unit_length / self.grain_size
        if f_gb_max > 1:
            warn('f_gb_max will be set to 1.')
            f_gb_max = 1
        return f_gb_max

    @property
    def name_fit_params(self):
        name = f"{self.chemical_composition.__str__()}_{int(self.temp_celsius)}C_d" \
               f"{np.round(self.grain_size * 1e6, 0)}micrometers"
        name = (name
                .replace('-', '_')
                .replace('.', 'p')
                )
        return name

    def plot(self):
        """Plot the existing simulated and/or experimental data.

        """
        fig, ax = plt.subplots()
        ax.plot(self.kinetics.experiment_time_datapoints, self.kinetics.experiment_fraction_datapoints,
                label='Experimental')
        ax.plot(self.kinetics.simulation_time_datapoints, self.kinetics.simulation_fraction_datapoints,
                label='Simulation')
        ax.set_xlim(0, np.max(np.concatenate((self.kinetics.experiment_time_datapoints,
                                              self.kinetics.simulation_time_datapoints))))
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fraction of bainite')
        ax.text(0, 1, f'{self.name}\n'
                      f'Q (GB) = {np.round(self.act_energy_gb/1000)} kJ/mol\n'
                      f'Q (A) = {np.round(self.act_energy_autocatalysis/1000)} kJ/mol\n'
                      f'Xb = {np.round(self.xb, 5)} at. fr.',
                va='top', ha='left')

    def print_parameters(self):
        """
        Print the activation energies for nucleation and the carbon content in bainite.

        """
        print(f'{self.name}\n--------------------------------\n'
              f'The activation energy for grain boundary nucleation (Q_GB) is:'
              f' {np.round(self.act_energy_gb/1000)} kJ/mol\n'
              f'The activation energy for autocatalytic nucleation (Q_A) is:'
              f' {np.round(self.act_energy_autocatalysis/1000)} kJ/mol\n'
              f'The atomic fraction of carbon trapped in the bainite (Xb) is: {np.round(self.xb, 5)}')

    def __str__(self):
        return self.name


def bainite_model(bainite_params: Bainite, *, act_energy_gb: float = None, act_energy_autocatalysis: float = None,
                  xb: float = None, time: np.ndarray = None, solver_options: dict = None):
    """
    Simulate the bainite fraction as a function of time.

    Parameters
    ----------
    bainite_params: Bainite
        Instance of bainite containing all the parameters related to the treatment.
    act_energy_gb: float, optional
        Activation energy for bainite nucleation at the austenite grain boundaries, in J/mol. If defined, it overrides
        the activation energy defined in bainite_params.
    act_energy_autocatalysis: float, optional
        Activation energy for bainite nucleation by autocalysis, in J/mol. If defined, it overrides the activation
        energy defined in bainite_params.
    xb: float, optional
        Carbon content trapped in the bainite, in atomic fraction. If defined, it overrides the xb defined
        in bainite_params.
    time: numpy.ndarray, optional
        Time datapoints for calculating the kinetics, in seconds. If defined, it overrides the time defined in
        bainite_params. If not defined, the array in bainite_params.kinetics.simulation_time_datapoints is used. If
        bainite_params.kinetics.simulation_time_datapoints is not defined, then the array in
        bainite_params.kinetics.experiment_fraction_datapoints is used.
    solver_options: dict
        Additional options to be passed to scipy.integrate.solve_ivp for solving the differential equation.

    Returns
    -------
    bainite_fraction: OdeResult
        ODE result from scipy.integrate.solve_ivp. Important attributes are:
        t : Time points
        y : Bainite fraction points in a (2,)-shaped array. The first value is the fraction of bainite
            formed by grain boundary nucleation and the second value is the fraction of bainite formed by autocatalysis.

    """
    # Define all necessary parameters
    q_gb = act_energy_gb if act_energy_gb is not None else bainite_params.act_energy_gb
    q_a = act_energy_autocatalysis if act_energy_autocatalysis is not None else bainite_params.act_energy_autocatalysis
    xb = xb if xb is not None else bainite_params.xb
    time = (time if time is not None
            else bainite_params.kinetics.simulation_time_datapoints if
            len(bainite_params.kinetics.simulation_time_datapoints) > 0
            else bainite_params.kinetics.experiment_time_datapoints)

    for key, val in {'act_energy_gb': q_gb, 'act_energy_autocatalysis': q_a, 'xb': xb}.items():
        if not isinstance(val, float) and not isinstance(val, int):
            raise ValueError(f"{val} is not a valid value for {key}")

    if solver_options is None:
        solver_options = {}

    grain_size = bainite_params.grain_size
    grain_shape = bainite_params.grain_shape
    n_s_gb = bainite_params.n_s_gb
    n_s_a = bainite_params.n_s_autocatalysis
    f_gb_max = bainite_params.f_gb_max
    sub_unit_length = bainite_params.sub_unit_length
    sub_unit_volume = bainite_params.sub_unit_volume
    x_avg = bainite_params.chemical_composition.c_at_fr
    c1 = bainite_params.chemical_composition.c1
    c2 = bainite_params.chemical_composition.c2
    k_gamma = bainite_params.chemical_composition.k_gamma
    t0_prime_avg = bainite_params.chemical_composition.t0prime_celsius
    temp_celsius = bainite_params.temp_celsius
    temp_kelvin = temp_celsius + 273
    vibration_factor = k * temp_kelvin / h

    # Differential equation
    def dfdt(f):
        """
        Calculate the rate of bainite formation at any given time.

        Parameters
        ----------
        f: numpy.ndarray
            Fraction of bainite at the given time in a (2,)-shaped array. The first value is the fraction of bainite
            formed by grain boundary nucleation and the second value is the fraction of bainite formed by autocatalysis.

        Returns
        -------
        df_dt: numpy.ndarray
            Rate of bainite formation at the given time in a (2,)-shaped array. The first value is the rate of bainite
            formation by grain boundary nucleation and the second value is the rate of bainite formation by
            autocatalysis.

        """
        f_gb = f[0]
        f_a = f[1]
        f_sum = f_gb + f_a

        # Carbon enrichment
        # t0_prime_avg is the T0' temperature, in Celsius degrees, at the beginning of the transformation, when the
        # carbon content in the austenite is equal to the average carbon content of the steel
        # t0_prime_t is the T0' temperature, in Celsius degrees, at time t, considering carbon enrichment
        # and q_gb_t and q_a_t follow the same convention
        t0_prime_t = t0_prime_avg - c2 * f_sum * (x_avg - xb) / (1 - f_sum)
        enrichment_factor = (t0_prime_t - temp_celsius) / (t0_prime_avg - temp_celsius)
        q_gb_t = q_gb + (k_gamma * c1 * f_sum * (x_avg - xb) / (1 - f_sum))
        q_a_t = q_a + (k_gamma * c1 * f_sum * (x_avg - xb) / (1 - f_sum))

        # Density of potential nucleation sites
        n_t_gb = n_s_gb * (grain_shape / grain_size) * (1 - f_gb / f_gb_max) * enrichment_factor
        f_a_max = 1 - f_gb_max ** (f_gb / f_gb_max)
        n_t_a = n_s_a * (f_sum / sub_unit_length) * (1 - f_a / f_a_max) * enrichment_factor if f_a_max > 0 else 0

        # Nucleation rates
        dn_gb_dt = vibration_factor * n_t_gb * np.exp(- q_gb_t / (r * temp_kelvin))
        dn_a_dt = vibration_factor * n_t_a * np.exp(- q_a_t / (r * temp_kelvin))

        # Rates of bainite formation
        df_gb_dt = dn_gb_dt * sub_unit_volume
        df_a_dt = dn_a_dt * sub_unit_volume
        df_dt = np.array([df_gb_dt, df_a_dt])

        return df_dt

    bainite_fraction = solve_ivp(fun=lambda t, f: dfdt(f), t_span=(time[0], time[-1]), y0=np.array([0, 0]),
                                 t_eval=time, **solver_options)

    # if len(bainite_fraction.y[0]) != len(time):
    #     f_gb_final = bainite_fraction.y[0][-1]
    #     f_a_final = bainite_fraction.y[1][-1]
    #     bainite_fraction.y[0] = np.concatenate(
    #         (bainite_fraction.y[0],
    #          f_gb_final * np.ones(len(time) - len(bainite_fraction.y[0]))
    #          )
    #     )
    #     bainite_fraction.y[1] = np.concatenate(
    #         (bainite_fraction.y[1],
    #          f_a_final * np.ones(len(time) - len(bainite_fraction.y[0]))
    #          )
    #     )

    return bainite_fraction


def simplified_bainite_model(bainite_params: Bainite, guessed_params):
    """
    Simulate the bainite fraction as a function of time using the analytical solution for a simplified version of the
    model.
    Intended use is only for fine-tuning initial guess of parameter values for fitting.

    """
    grain_size = bainite_params.grain_size
    grain_shape = bainite_params.grain_shape
    n_s_gb = bainite_params.n_s_gb
    n_s_a = bainite_params.n_s_autocatalysis
    sub_unit_length = bainite_params.sub_unit_length
    sub_unit_volume = bainite_params.sub_unit_volume
    x_avg = bainite_params.chemical_composition.c_at_fr
    c2 = bainite_params.chemical_composition.c2
    t0_prime_avg = bainite_params.chemical_composition.t0prime_celsius
    temp_celsius = bainite_params.temp_celsius
    temp_kelvin = temp_celsius + 273
    vibration_factor = k * temp_kelvin / h
    time = bainite_params.kinetics.experiment_time_datapoints

    q_gb = guessed_params['act_energy_gb'].value
    q_a = guessed_params['act_energy_autocatalysis'].value
    xb = guessed_params['xb'].value

    nuc_sites_gb = n_s_gb * (grain_shape / grain_size)
    autocatalytic_factor = ((n_s_a / n_s_gb) * grain_size / (grain_shape * sub_unit_length)
                            * np.exp((q_gb - q_a) / (r * temp_kelvin)))
    kappa = vibration_factor * nuc_sites_gb * np.exp(- q_gb / (r * temp_kelvin)) * sub_unit_volume
    max_c_aus = x_avg + (t0_prime_avg - temp_kelvin) / c2
    f_max = (max_c_aus - x_avg) / (max_c_aus - xb)
    fraction = (f_max * (1 - np.exp(-kappa * (1 + autocatalytic_factor) * time))
                / (autocatalytic_factor * np.exp(-kappa * (1 + autocatalytic_factor) * time) + 1))

    return fraction


def make_single_fit_params(bainite_params: Bainite) -> Parameters:
    """
    Create and estimate parameters for fitting single curve of bainite formation.
    The estimate is based on fitting a simplified version of the model, simplified_bainite_model().

    """
    # Activation energies
    q_gb_guess = 114e3 + 308 * bainite_params.temp_celsius
    q_a_guess = 123e3 + 269 * bainite_params.temp_celsius

    # Carbon trapped in bainite
    n_datapoints = len(bainite_params.kinetics.experiment_fraction_datapoints)
    c_at_fr_avg = bainite_params.chemical_composition.c_at_fr
    t0prime_celsius = bainite_params.chemical_composition.t0prime_celsius
    temp_celsius = bainite_params.temp_celsius
    delta_temp = t0prime_celsius - temp_celsius
    c2 = bainite_params.chemical_composition.c2

    if n_datapoints > 100:
        final_f = np.average(bainite_params.kinetics.experiment_fraction_datapoints[-int(np.round(n_datapoints*0.01,
                                                                                                  0))])
    else:
        final_f = bainite_params.kinetics.experiment_fraction_datapoints[-1]

    xb_guess = c_at_fr_avg - delta_temp * (1 - final_f) / (c2 * final_f)

    if xb_guess > c_at_fr_avg:
        xb_guess = c_at_fr_avg

    # Create parameters
    fit_params = Parameters()
    fit_params.add('act_energy_gb', value=q_gb_guess, min=q_gb_guess*0.5, max=q_gb_guess*1.5)
    fit_params.add('act_energy_autocatalysis', value=q_a_guess, min=q_a_guess*0.5, max=q_a_guess*1.5)
    fit_params.add('xb', value=xb_guess, min=0, max=c_at_fr_avg)

    # Fine-tune parameters
    def residuals_simplified_model(guessed_params):
        experiment = bainite_params.kinetics.experiment_fraction_datapoints
        model = simplified_bainite_model(bainite_params, guessed_params)
        return experiment - model

    fine_tuned_params = minimize(residuals_simplified_model, fit_params)

    q_gb_guess = fine_tuned_params.params['act_energy_gb'].value
    q_a_guess = fine_tuned_params.params['act_energy_autocatalysis'].value
    xb_guess = fine_tuned_params.params['xb'].value
    fit_params = Parameters()
    fit_params.add('act_energy_gb', value=q_gb_guess, min=q_gb_guess*0.8, max=q_gb_guess*1.2)
    fit_params.add('act_energy_autocatalysis', value=q_a_guess, min=q_a_guess*0.8, max=q_a_guess*1.2)
    fit_params.add('xb', value=xb_guess, min=0, max=c_at_fr_avg)

    return fit_params


def fit_single_curve(bainite_params: Bainite, *, fit_params: Parameters = None, update_bainite_params: bool = True,
                     solver_options: dict = None, minimizer_options: dict = None) -> minimizer.MinimizerResult:
    """
    Fit an experimental curve of bainite formation to the model to find the optimized values of the activation
    energies and the carbon content in the bainite.

    Parameters
    ----------
    bainite_params: Bainite
        Should contain the necessary parameters for the model (except the parameters to be fitted) and the
        experimental data.
    fit_params: Parameters, optional
        If defined, fit_params is used as the params argument of the lmfit.minimize function. This parameter is useful
        for holding constant one or more parameters to be fitted, or to use different starting values for the
        optimization.
        The Parameters instanced should contain:
                'act_energy_gb': Activation energy for bainite nucleation at the austenite grain boundaries, in J/mol.
                'act_energy_autocatalysis': Activation energy for bainite nucleation by autocalysis, in J/mol.
                'xb': Carbon content trapped in the bainite, in atomic fraction.
        If None, the parameters will be automatically created and initial values will be estimated using the
        make_single_fit_params function.
        Default is None.
    update_bainite_params: bool, optional
        If True, the fitted values will be passed on the bainite_params. Default is True.
    solver_options: dict, optional
        Additional options to be passed to scipy.integrate.solve_ivp for solving the differential equation.
    minimizer_options: dict, optional
        Additional options to be passed to lmfit.minimize for fitting.

    Returns
    -------
    fit: lmfit.minimizer.MinimizeResult
        Output result of the lmfit.minimize function containing the fitted parameters and goodness-of-fit
        statistics.

    """
    if (bainite_params.kinetics.experiment_time_datapoints == np.array([]) or
            bainite_params.kinetics.experiment_fraction_datapoints == np.array([])):
        raise AttributeError(f"Missing experimental datapoints in {bainite_params}.kinetics.")
    if (len(bainite_params.kinetics.experiment_time_datapoints)
            != len(bainite_params.kinetics.experiment_fraction_datapoints)):
        raise IndexError(f"Time and fraction arrays in {bainite_params}.kinetics should have same length.")

    if fit_params is None:
        fit_params = make_single_fit_params(bainite_params)

    if solver_options is None:
        solver_options = {}

    if minimizer_options is None:
        minimizer_options = {}

    def residuals(params):
        q_gb = params['act_energy_gb'].value
        q_a = params['act_energy_autocatalysis'].value
        xb = params['xb'].value
        time = bainite_params.kinetics.experiment_time_datapoints
        experiment = bainite_params.kinetics.experiment_fraction_datapoints
        model_ode = bainite_model(bainite_params, act_energy_gb=q_gb, act_energy_autocatalysis=q_a, xb=xb, time=time,
                                  solver_options=solver_options)
        model = model_ode.y[0] + model_ode.y[1]
        return experiment - model

    fit = minimize(residuals, fit_params, **minimizer_options)

    if update_bainite_params is True:
        bainite_params.act_energy_gb = fit.params['act_energy_gb'].value
        bainite_params.act_energy_autocatalysis = fit.params['act_energy_autocatalysis'].value
        bainite_params.xb = fit.params['xb'].value
        simulated_curve = bainite_model(bainite_params)
        bainite_params.kinetics.simulation_time_datapoints = simulated_curve.t
        bainite_params.kinetics.simulation_fraction_datapoints = simulated_curve.y[0] + simulated_curve.y[1]

    return fit


def make_multiple_fit_params(bainite_params: [Bainite], share_q_gb: bool, share_q_a: bool,
                             share_xb: bool) -> Parameters:
    """
    Create and estimate parameters for fitting multiple curves of bainite formation.
    The estimate is based on fitting a simplified version of the model, simplified_bainite_model().

    """
    fit_params_combined = Parameters()

    for bain in bainite_params:
        params = make_single_fit_params(bain)
        act_energies = (param for param in params.values() if param.name.startswith('act_'))
        for param in act_energies:
            fit_params_combined.add(f'{param.name}_{bain.name_fit_params}',
                                    value=param.value, min=param.value * 0.8, max=param.value * 1.2)
        fit_params_combined.add(f'xb_{bain.name_fit_params}', value=params['xb'].value,
                                min=0, max=bain.chemical_composition.c_at_fr)

    if share_q_gb is True:
        for bain in bainite_params[1:]:
            fit_params_combined[f'act_energy_gb_{bain.name_fit_params}'].value = (
                fit_params_combined[f'act_energy_gb_{bainite_params[0].name_fit_params}'].value
            )
            fit_params_combined[f'act_energy_gb_{bain.name_fit_params}'].expr = (
                f'act_energy_gb_{bainite_params[0].name_fit_params}'
            )

    if share_q_a is True:
        for bain in bainite_params[1:]:
            fit_params_combined[f'act_energy_autocatalysis_{bain.name_fit_params}'].value = (
                fit_params_combined[f'act_energy_autocatalysis_{bainite_params[0].name_fit_params}'].value
            )
            fit_params_combined[f'act_energy_autocatalysis_{bain.name_fit_params}'].expr = (
                f'act_energy_autocatalysis_{bainite_params[0].name_fit_params}'
            )

    if share_xb is True:
        for bain in bainite_params[1:]:
            fit_params_combined[f'xb_{bain.name_fit_params}'].value = (
                fit_params_combined[f'xb_{bainite_params[0].name_fit_params}'].value
            )
            fit_params_combined[f'xb_{bain.name_fit_params}'].expr = (
                f'xb_{bainite_params[0].name_fit_params}'
            )

    return fit_params_combined


def fit_multiple_curves(bainite_params: list[Bainite], *, fit_params: Parameters = None, share_q_gb: bool = True,
                        share_q_a: bool = True, share_xb: bool = False, update_bainite_params: bool = True,
                        solver_options: dict = None, minimizer_options: dict = None):
    """
    Fit multiple experimental curves of bainite formation to the model to find the optimized values of the activation
    energies and the carbon content in the bainite.


    Parameters
    ----------
    bainite_params: list[Bainite]
        List containing instances of Bainite to be fitted. Each instance should contain the necessary parameters for
        the model (except the parameters to be fitted) and the experimental data.
    fit_params: Parameters, optional
        If defined, fit_params is used as the params argument of the lmfit.minimize function. This parameter is useful
        for holding constant one or more parameters to be fitted, or to use different starting values for the
        optimization.
        The Parameters instanced should contain the parameters for all the instances of the bainite_params list,
        and should be named as:
                'act_energy_gb_{bainite_params.name}': Activation energy for bainite nucleation at the austenite
                 grain boundaries, in J/mol.
                'act_energy_autocatalysis_{bainite_params.name}': Activation energy for bainite nucleation by
                 autocalysis, in J/mol.
                'xb_{bainite_params.name}': Carbon content trapped in the bainite, in atomic fraction.
        If None, the parameters will be automatically created and initial values will be estimated using the
        make_single_fit_params function.
        Default is None.
    share_q_gb: bool
        If True, all curves will share the same activation energy for grain boundary nucleation during fitting.
        Default is True.
    share_q_a: bool
        If True, all curves will share the same activation energy for autocatalytic nucleation during fitting.
        Default is True.
    share_xb: bool
        If True, all curves will share the same carbon content trapped in the bainite. Default is False.
    update_bainite_params: bool
        If True, the fitted values will be passed on the bainite_params. Default is True.
    solver_options: dict, optional
        Additional options to be passed to scipy.integrate.solve_ivp for solving the differential equation.
    minimizer_options: dict, optional
        Additional options to be passed to lmfit.minimize for fitting.

    Returns
    -------
    fit: lmfit.minimizer.MinimizeResult
        Output result of the lmfit.minimize function containing the fitted parameters and goodness-of-fit
        statistics.

    """
    if len(bainite_params) <= 1:
        raise IndexError('bainite_params should contain at least two elements')
    
    for bain in bainite_params:
        if (bain.kinetics.experiment_time_datapoints == np.array([]) or
                bain.kinetics.experiment_fraction_datapoints == np.array([])):
            raise AttributeError(f"Missing experimental datapoints in {bain}.kinetics.")
        if (len(bain.kinetics.experiment_time_datapoints)
                != len(bain.kinetics.experiment_fraction_datapoints)):
            raise IndexError(f"Time and fraction arrays in {bain}.kinetics should have same length.")

    if fit_params is None:
        fit_params = make_multiple_fit_params(bainite_params, share_q_gb=share_q_gb,
                                              share_q_a=share_q_a, share_xb=share_xb)

    if solver_options is None:
        solver_options = {}

    if minimizer_options is None:
        minimizer_options = {}

    def residuals(parameters) -> np.ndarray:
        full_residuals = np.array([])
        for bainite_param in bainite_params:
            q_gb = parameters[f'act_energy_gb_{bainite_param.name_fit_params}'].value
            q_a = parameters[f'act_energy_autocatalysis_{bainite_param.name_fit_params}'].value
            xb = parameters[f'xb_{bainite_param.name_fit_params}'].value
            model = bainite_model(bainite_param, act_energy_gb=q_gb, act_energy_autocatalysis=q_a, xb=xb,
                                  solver_options=solver_options)
            model_fraction = model.y[0] + model.y[1]
            experimental = bainite_param.kinetics.experiment_fraction_datapoints
            if model_fraction.shape != experimental.shape:
                print(f"{q_gb}, {q_a}, {xb}")
            residual = experimental - model_fraction
            residual_weighted = residual / np.sqrt(len(experimental))
            full_residuals = np.concatenate((full_residuals, residual_weighted))

        return full_residuals

    fit = minimize(residuals, fit_params, **minimizer_options)

    if update_bainite_params is True:
        for bainite in bainite_params:
            bainite.act_energy_gb = fit.params[f'act_energy_gb_{bainite.name_fit_params}'].value
            bainite.act_energy_autocatalysis = fit.params[f'act_energy_autocatalysis_{bainite.name_fit_params}'].value
            bainite.xb = fit.params[f'xb_{bainite.name_fit_params}'].value
            simulated_curve = bainite_model(bainite)
            bainite.kinetics.simulation_time_datapoints = simulated_curve.t
            bainite.kinetics.simulation_fraction_datapoints = simulated_curve.y[0] + simulated_curve.y[1]

    return fit
