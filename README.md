# About
Code accompanying the paper "Modeling the effect of prior austenite grain size on bainite formation kinetics."

# Examples
This repository contains three examples that are sufficient for replicating all the results presented in the original publication.

[Example 1](Example1.py) shows how to simulate and plot a curve of bianite formation given the steel's chemical composition, isothermal holding temperature, activation energies for nucleation and carbon content in bainite.

[Example 2](Example2.py) shows how to fit experimental data from a single curve of bainite formation to the model and extract fitted values of the activation energies for nucleation and the carbon content in bainite.

[Example 3](Example3.py) shows how to fit to the model experimental data from multiple curves of bainite formation for a given chemical composition and different prior austenite grain sizes and extract the activation energies for nucleation, which are shared among the steels with different prior austenite grain sizes, and the carbon content in bainite.

# Required libraries
The code uses the following python libraries:

-   [numpy](https://numpy.org/)
-   [matplotlib](https://matplotlib.org/)
-   [scipy](https://scipy.org/)
-   [periodictable](https://periodictable.readthedocs.io/en/latest/)
-   [lmfit](https://lmfit.github.io/lmfit-py/)
-   [pandas](https://pandas.pydata.org/)
