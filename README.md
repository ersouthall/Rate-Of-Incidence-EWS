# Rate-Of-Incidence-EWS
Code to accompany "Prospects for detecting early warning signals in discrete event sequence data: application to epidemiological incidence data."


## Abstract
Early warning signals (EWS) identify systems approaching a critical transition, where the system undergoes a sudden change in state. For example, monitoring changes in variance or autocorrelation offers a computationally inexpensive method which can be used in real-time to assess when an infectious disease transitions to elimination.
EWS have a promising potential to not only be used to monitor infectious diseases, but also to inform control policies to aid disease elimination. Previously, potential EWS have been identified for prevalence data, however the prevalence of a disease is often not known directly. In this work we identify EWS for incidence data, the standard data type collected by the Centers for Disease Control and Prevention (CDC) or World Health Organization (WHO). We show, through several examples, that EWS calculated on simulated incidence time series data exhibit vastly different behaviours to those previously studied on prevalence data. In particular, the variance displays a decreasing trend on the approach to disease elimination, contrary to that expected from critical slowing down theory; this could lead to unreliable indicators of elimination when calculated on real-world data.  We derive analytical predictions which can be generalised for many epidemiological systems, and we support our theory with simulated studies of disease incidence.  Additionally, we explore EWS calculated on the rate of incidence over time, a property which can be extracted directly from incidence data. We find that although incidence might not exhibit typical critical slowing down properties before a critical transition, the rate of incidence does, presenting a promising new data type for the application of statistical indicators.


## Usage

### Simulations
This repository has the simulation scripts which were used to get the simulated data in the manuscript.
 - Python code to reproduce these simulations can be found in "simulations/python_files/"
- Bash scripts are provided to run the Python files in "simulations/bash_scripts/"
- Example output data after running 500 realisations in parallel on a HPC can be found "data/dataHPC/"

### Incidence, Prevalence and Rate of Incidence 
Functions contained in `functions/simulation_output_funcs.py` process the Gillespie output data and evaluate the corresponding theoretical solutions (e.g. the mean-field equations for infecteds and susceptibles, the RoI SDE and the mean of the Poisson Process). 

In particular, `RoI_approximation(..)` calculates the Rate of Incidence from prevalence output ("true RoI", method 1) and on a rolling window from new cases output ("rolling RoI", method 2).

### Statisitcal Indicators
Python functions of the EWS considered in the paper and supplementary material (variance, coefficient of variation, autocorrelation lag-1, skewness and kurtosis) can be found in "functions/stats_funcs.py". 

Functions are provided for calculating EWS on the simulated data and theoretical functions (ode solver or integration) in:  `functions/stats_funcs.py`

Example notebooks which evaluate the EWS functions and then plots them (using `functions/plots_func.py` is given in `notebooks`. This is broken down by figure (or supplementary figure) in the manuscript.
