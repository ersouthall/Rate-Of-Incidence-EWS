# gillespie run, SIS no birth, deaths or external

import sys
sys.path.append('../../functions')


from simulation_funcs import gillespieVaccExt, gillespieVaccFix, gillespieSteps
from multiprocessing import Pool
import numpy as np 
pyfile = sys.argv[0]
populationN = int(sys.argv[1])
beta0 = float(sys.argv[2])
gamma = float(sys.argv[3])
p0 = float(sys.argv[4])
mu = float(sys.argv[5])
simType = sys.argv[6]
repeats = int(sys.argv[7])
#print(sys.argv)
print(populationN, beta0, gamma, simType)

# other parameters
pRate =1/500
changeTime = 500
burnTime = 300

def para(gillespie_algorithm):
    gill = gillespie_algorithm(initial = [0.2*populationN,
                                           0.8*populationN, 0],
				p0 = p0,
                                beta0=beta0,
                                p=pRate,
                                gamma=gamma,
				mu=mu,
                                max_time=changeTime,
                                burntime=burnTime)
#     gill = gillespieSISFix([0.2*N, 0.8*N], β_0,p, γ, T,BT)
#     gill = gillespieSISNExt([0.2*N, 0.8*N], β_0,p, γ, T, (1-1.3*γ)/p, BT)
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
#realisations = 500
if simType == 'Ext':
    print('Extinction run')
    functionInput = gillespieVaccExt
elif simType == 'Fix':
    print('Steady state run')
    functionInput = gillespieVaccFix
else:
    print('incorrect simulation type inputted (argument 4), choose Ext or Fix')

print('\n Starting parallel runs')
print('running ', repeats, ' realisations')

runs = [functionInput for i in range(repeats)]
num_threads = 12

with Pool(num_threads) as pool:
    results = pool.map(para, runs)

print('\n simulation runs complete')

print('\n saving...')

#input user directory 
# usr_dir = '/home/usr/'
#input where to save the output
# incidence_dir = usr_dir + 'Documents/save/npyFiles/'

np.save(incidence_dir+'Nlarge/'+simType+'_SIS_vaccination.npy', results)
