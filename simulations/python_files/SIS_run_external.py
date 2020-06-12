# gillespie run, SIS with external infections


import sys
sys.path.append('../../functions')

from simulation_funcs import gillespieSISEm, gillespieSISFix, gillespieSteps
from multiprocessing import Pool
import numpy as np 

pyfile = sys.argv[0]
populationN = int(sys.argv[1])
beta0 = float(sys.argv[2])
gamma = float(sys.argv[3])
simType = sys.argv[4]
realisations = int(sys.argv[5])

# other parameters
pRate =1/500
changeTime = 500
burnTime = 300
nu = 0.001


def para(gillespie_algorithm):
    gill = gillespie_algorithm(initial = [populationN,
                                           0],
                                beta0=beta0,
                                p=pRate,
                                gamma=gamma,
                                max_time=changeTime,
                                burntime=burnTime, 
                                nu = nu)

    steps = gillespieSteps(gillespieOuput=gill,
                           T = changeTime, 
                           BT = burnTime)
    inter_t = np.arange(0, round(max(steps[0]))+1, 0.1)
    inter_i = np.interp(inter_t, steps[0],steps[2])
    inter_s = np.interp(inter_t, steps[0], steps[1])
    inter_i = inter_i[:(changeTime+burnTime)*10]
    inter_s = inter_s[:(changeTime+burnTime)*10]
    inter_NC = steps[3]
    return inter_i, inter_s, inter_NC

# embarrasingly parallel
print('parallel inputs:')
if simType == 'Em':
    print('Extinction run')
    functionInput = gillespieSISEm
elif simType == 'Fix':
    print('Steady state run')
    functionInput = gillespieSISFix
else:
    print('incorrect simulation type inputted (argument 4), choose Em or Fix')

print('\n Starting parallel runs')

runs = [functionInput for i in range(realisations)]
num_threads = 12

with Pool(num_threads) as pool:
    results = pool.map(para, runs)
    
print('\n simulation runs complete')

print('\n saving...')

#input user directory 
# usr_dir = '/home/usr/'
#input where to save the output
# incidence_dir = usr_dir + 'Documents/save/npyFiles/'

np.save(incidence_dir+'Nlarge/'+simType+'_SIS_emergence.npy',results)

