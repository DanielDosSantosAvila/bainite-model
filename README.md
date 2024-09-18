# About
Code accompanying the paper:
dos Santos Avila, D., Offerman, S. E., & Santofimia, M. J. (2024).
Modeling the effect of prior austenite grain size on bainite formation kinetics.
Acta Materialia, 266, 119656.
doi.org/10.1016/j.actamat.2024.119656

If you need any assistance, email me at D.dosSantosAvila@tudelft.nl

# Examples
This repository contains three examples that are sufficient for replicating all the results presented in the original publication.

[Example 1](Example1.py) shows how to simulate and plot a curve of bianite formation given the steel's chemical composition, isothermal holding temperature, activation energies for nucleation and carbon content in bainite.

[Example 2](Example2.py) shows how to fit experimental data from a single curve of bainite formation to the model and extract fitted values of the activation energies for nucleation and the carbon content in bainite. For better results, use [Example2_interactive.py](Example2_interactive.py)

[Example 3](Example3.py) shows how to fit experimental data from multiple curves of bainite formation for a given chemical composition and different prior austenite grain sizes and extract the activation energies for nucleation, which are shared among the steels with different prior austenite grain sizes, and the carbon content in bainite. For better results, use [Example3_interactive.py](Example3_interactive.py)

When you use run the interactive examples, an interactive matplotlib windows should open (see below). The simulated kinetics of bainite formation should be automatically updated as the values of the slider are changed.

![Interactive](https://github.com/user-attachments/assets/2f892aa5-4a4f-4a3e-843a-99adbfe4e5f7)

# Required libraries
The code is running on Python 3.11 and uses the following Python libraries:

-   [numpy](https://numpy.org/)
-   [matplotlib](https://matplotlib.org/)
-   [scipy](https://scipy.org/)
-   [periodictable](https://periodictable.readthedocs.io/en/latest/)
-   [lmfit](https://lmfit.github.io/lmfit-py/)
-   [pandas](https://pandas.pydata.org/)
