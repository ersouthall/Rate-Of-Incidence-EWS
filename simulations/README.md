# Run Stochastic Simulations

Run the simulations using the Gillespie algorithm with the following: 

- Run elimination model with social distancing:

`cd bash_scripts
source SIS_run_beta_elimination.sh`

- Run elimination model with vaccination:

`cd bash_scripts
source SIS_run_vacc_elimination.sh`

- Run emergence model:

`cd bash_scripts
source SIS_run_external_emergence.sh`

Edit the bash scripts to change the number of realisations (set at 500) or  parameter values (e.g. population size, initial beta, gamma, mu, nu) 
 
