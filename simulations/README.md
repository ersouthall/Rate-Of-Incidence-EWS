[![Python 3.7.7](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


# Run Stochastic Simulations

Run the simulations using the Gillespie algorithm with the following: 

- Run elimination model with social distancing:
```
$ cd bash_scripts
$ source SIS_run_beta_elimination.sh
```
- Run elimination model with vaccination:
```
$ cd bash_scripts
$ source SIS_run_vacc_elimination.sh
```
- Run emergence model:
```
$ cd bash_scripts
$ source SIS_run_external_emergence.sh
```
Edit the bash scripts to change the number of realisations (set at 500) or  parameter values (e.g. population size, initial beta, gamma, mu, nu) 

The `requirements.txt` file lists all Python libraries that the notebooks depend on.
