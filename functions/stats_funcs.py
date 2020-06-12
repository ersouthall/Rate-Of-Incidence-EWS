import numpy as np
from scipy.integrate import odeint
from output_funcs import NC 
from scipy.stats import skew, kurtosis
from scipy.integrate import quad

#################################
# EWS calculated on simulations:
# variance
# coeficent of variation
# skewness 
# kurtosis
# autocorrelation lag tau 
#################################

def variance_simulations(results, data_type, param_dict):
    if data_type == 'Incidence':
        simulation = np.var(results['I_sim'],0)/param_dict['N']
    elif data_type=='Prevalence':
        simulation = np.var(results['P_sim'],0)/param_dict['N']
    else:
        simulation = np.var(results,0)/param_dict['N']
    return simulation

def CoV_simulations(results, data_type, param_dict):
    if data_type == 'Incidence':
        detrend = results['I_sim']
#         detrend = results['I_sim'] - np.mean(results['I_sim'], 0)
        simulation = np.std(detrend,0)/np.mean(detrend,0)
    elif data_type=='Prevalence':
        detrend = results['P_sim']
#         detrend = results['P_sim'] - np.mean(results['P_sim'], 0)
        simulation = np.std(detrend,0)/np.mean(detrend,0)
    else:
        detrend = results
#         detrend = results- np.mean(results, 0)
        simulation = np.std(detrend,0)/np.mean(detrend,0)
    return simulation

def Skew_simulations(results, data_type, param_dict):
    if data_type == 'Incidence':
        detrend = results['I_sim'] - np.mean(results['I_sim'], 0)
        simulation = skew(detrend,0)
    elif data_type=='Prevalence':
        detrend = results['P_sim'] - np.mean(results['P_sim'], 0)
        simulation = skew(detrend,0)
    else:
        detrend = results- np.mean(results, 0)
        simulation = skew(detrend,0)
    return simulation

def Kurt_simulations(results, data_type, param_dict):
    if data_type == 'Incidence':
        detrend = results['I_sim'] - np.mean(results['I_sim'], 0)
        simulation = kurtosis(detrend,0)
    elif data_type=='Prevalence':
        detrend = results['P_sim'] - np.mean(results['P_sim'], 0)
        simulation = kurtosis(detrend,0)
    else:
        detrend = results- np.mean(results, 0)
        simulation = kurtosis(detrend,0)
    return simulation

def AC_simulations(results, data_type, param_dict):
    if data_type == 'Incidence':
        detrend = results['I_sim'] - np.mean(results['I_sim'], 0)
        simulation = autocorrelation_tau(detrend,param_dict)
    elif data_type=='Prevalence':
        detrend = results['P_sim'] - np.mean(results['P_sim'], 0)
        daily_detrend = detrend[:, np.arange(0, (param_dict['T']+param_dict['BT'])*10, 10)]
        simulation = autocorrelation_tau(daily_detrend,param_dict)
    else:
        detrend = results- np.mean(results, 0)
        try:
            daily_detrend = detrend[:, np.arange(0, (param_dict['T']+param_dict['BT'])*10, 10)]
        except: 
            daily_detrend = detrend
        simulation = autocorrelation_tau(daily_detrend,param_dict)
    return np.mean(simulation, 0 )

def autocorrelation_tau(detrended_data, param_dict):
    '''
    Function for calculating rolling autocorrelation lag-tau
    value of tau and window size are contained in param_dict
    
    Can use inbuilt function from Pandas:  pd.DataFrame(xtest.T).apply(lambda x:  pd.Series(x).autocorr(lag = 1)).values 
    but is significantly slower
    '''
    AC_results = np.zeros((param_dict['repeats'], param_dict['T']+param_dict['BT']))
    for i in (range(param_dict['windowSize']//2,param_dict['T']+param_dict['BT']-(param_dict['windowSize']//2))):
        if param_dict['windowSize']%2==0:
            subtract = 1
        else:
            subtract = 2
        data_window = detrended_data[:,( i-(param_dict['windowSize']//2)):(i+(param_dict['windowSize']//2))]
        x = data_window[:, :-1] -np.transpose((np.mean(data_window[:,:-param_dict['lag_tau']],
                                                 axis = 1),)*(param_dict['windowSize']-subtract
                                                                ))
        x_tau = data_window[:,1:] -  np.transpose((np.mean(data_window[:,param_dict['lag_tau']:],
                                                        axis = 1),)*(param_dict['windowSize']-subtract
                                                                ))
        numerator = np.mean(x*x_tau, axis = 1)
        denominator = np.std(x,axis =1)*np.std(x_tau, axis = 1)
        xtest_autocorr=numerator/denominator
        AC_results[:,i] = xtest_autocorr
    return AC_results 


##############################################
# EWS theory for incidence, prevalence and RoI:
# variance
# coeficent of variation
# skewness 
# kurtosis
# autocorrelation lag tau 
###############################################

def Kurt_theory(results_dict, data_type, RUN, model_no,
                        param_all, param_dict, theory_func= None, odea_theory = False):
    statistic_dict = {}
    ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
    if data_type == 'Incidence':
        
        theory = 1/np.sqrt(results_dict['I_theory'][:,0]*param_dict['N'])
        theory_col = 'tab:green'
        ts = ts
        theory_name = '(Poisson)'

        
    elif data_type == 'Prevalence':
        if model_no == 2:
            theory=np.sqrt(results_dict['P_theory'][:,4]/param_dict['N'])/(results_dict['P_theory'][:,1])
        else:
            theory = (results_dict['P_theory'][:,2])/(results_dict['P_theory'][:,1]**2) - 3
         
        theory_col = 'tab:red'
        ts = ts
        theory_name = '(SDE)'
    elif data_type == 'RoI':
        if model_no ==2:
            var_theory = theory_func(results_dict['P_theory'], param_dict['beta0'])
                                   
            theory = np.sqrt(var_theory[1]/param_dict['N'])/var_theory[0]
        else:
            phi_var_theory = odeint(theory_func, param_dict['I0'],ts, 
                 args = (RUN, param_all['BT'], param_dict['beta0'],
                          param_dict['gamma'],
                          param_dict['nu'],param_all['p'],
                          param_dict['psign'], ))
            mean_theory = NC(phi_var_theory, ts, param_all['BT'], 
                             0,param_all['T'], RUN, param_dict['beta0'], param_dict['gamma'],
                             param_all['p'],
                          param_dict['psign'])
            theory = np.sqrt(phi_var_theory[:,1]/param_dict['N'])/(mean_theory[:,0])
        theory_col = 'tab:orange'
        theory_name = '(SDE)'
    else:
        print('data-type name undefined')
    statistic_dict['theory'] = theory
    statistic_dict['theory_col'] = theory_col
    statistic_dict['ts'] = ts
    statistic_dict['theory_name'] = theory_name
    return statistic_dict

def betaChange(param_dict, param_all, RUN):
    betaFix = param_dict['beta0']*np.ones((param_all['T']+param_all['BT'])*10)
    betaExt =np.concatenate((param_dict['beta0']*np.ones(param_all['BT']*10),
                             param_dict['beta0']*(1+param_dict['psign']*param_all['p']*(np.arange(param_all['BT'],
                                                                                                    param_all['BT']+param_all['T'],
                                                                                                     0.1)-param_all['BT']))))
    betaNExt = np.concatenate((betaExt[:(param_all['BT']+param_all['ST'])*10],
                                1.3*param_dict['gamma']*np.ones((param_all['T']-param_all['ST'])*10))) 
    if RUN =='Fix':
        betaT = betaFix
    if RUN == 'Ext':
        betaT = betaExt
    if RUN == 'NExt':
        betaT = betaNExt
    return betaT

def PowerSpectrum_prevalence(omega, A, B, tau, param_all = None, param_dict = None, results_dict = None, timeindex = None):
    det_A = np.linalg.det(A)
    trace_A = np.trace(A)
    numerator = (A[1,0]**2)*B[0,0] - 2*A[1,0]*A[0,0]*B[0,1] + (A[0,0]**2)*B[1,1] + B[1,1]*(omega**2)
    denominator = (det_A - omega**2)**2 + (trace_A**2)*(omega**2)
    return np.cos(omega*tau)*(numerator/denominator )

def PowerSpectrum_RoI(omega, A, B, tau, param_all, param_dict, results_dict, timeindex):
    phi = results_dict['P_theory'][np.arange(0, (param_all['T']+param_all['BT'])*10, 10), 1][timeindex]
    psi = results_dict['P_theory'][np.arange(0, (param_all['T']+param_all['BT'])*10, 10), 0][timeindex]
    
    det_A = np.linalg.det(A)
    trace_A = np.trace(A)
    frac = (param_dict['beta0']**2)/((det_A - omega**2)**2 + (trace_A**2)*(omega**2))
    b11_coef = (A[1,1]**2)*(phi**2) - 2*A[1,0]*A[1,1]*phi*psi + (A[1,0]**2)*(psi**2)
    b22_coef = (A[0,1]**2)*(phi**2) - 2*A[0,1]*A[0,0]*phi*psi + (A[0,0]**2)*(psi**2)
    b12_coef = 2*((A[0,1]*A[1,0]+A[1,1]*A[0,0])*phi*psi - A[1,1]*A[0,1]*(phi**2)-A[1,0]*A[0,0]*(psi**2))
    omega_sq_coef = (phi**2)*B[0,0] + (psi**2)*B[1,1] + 2*phi*psi*B[0,1]
    
    result = frac*(b11_coef*B[0,0]+b22_coef*B[1,1]+b12_coef*B[0,1]+omega_sq_coef*(omega**2))
    return np.cos(omega*tau)*result

def autocorrelation_from_powerspectrum(results_dict, param_all, param_dict, RUN,
                                       matrices_func, powerspectrum_func):
    autocorr = []
    for index_time, time in enumerate(np.arange(0, param_all['T']+param_all['BT'], 1)):
        newrate = rateVaccineChange(time= time,
                                    param_all=param_all,
                                    param_dict=param_dict,
                                    RUN=RUN)

        A, B = matrices_func(results_dict=results_dict,
                                    param_dict=param_dict,
                                    param_all=param_all,
                                    RUN=RUN,
                                    vaccine_rate_pT=newrate,
                                    timeindex=index_time)


        ######### Integration #########


        var = (1/(2*np.pi))*quad(powerspectrum_func,
                                 -np.inf,
                                 np.inf,
                                 args=(A, B,0,param_all, param_dict, results_dict,index_time,  ))[0]
        c_tau_1 = (1/(2*np.pi))*quad(powerspectrum_func,
                                     -np.inf,
                                     np.inf,
                                     args=(A, B,1, param_all, param_dict, results_dict,index_time, ))[0]

        autocorr.append(c_tau_1/var)
    return autocorr

def matrixVaccineModel(results_dict, param_dict, param_all, RUN, vaccine_rate_pT, timeindex):
    phi = results_dict['P_theory'][np.arange(0, (param_all['T']+param_all['BT'])*10, 10), 1][timeindex]
    psi = results_dict['P_theory'][np.arange(0, (param_all['T']+param_all['BT'])*10, 10), 0][timeindex]

    A= np.matrix([[-param_dict['beta0']*phi - param_dict['mu'],
                   -param_dict['beta0']*psi+param_dict['gamma']],
                  [param_dict['beta0']*phi,
                   param_dict['beta0']*psi - param_dict['mu'] - param_dict['gamma']]])
    B = np.matrix([[param_dict['beta0']*phi*psi + param_dict['mu']*(1-psi) 
                    +param_dict['mu']*vaccine_rate_pT +param_dict['gamma']*phi,
                    -phi*(param_dict['beta0']*psi + param_dict['mu']+param_dict['gamma'])],
                   [-phi*(param_dict['mu']+param_dict['gamma']+param_dict['beta0']*psi),
                    phi*(param_dict['beta0']*psi + param_dict['mu']+param_dict['gamma'])]])
    return A, B

def rateVaccineChange(time, param_all,param_dict, RUN):
    if (time <param_all['BT'] or RUN=='Fix'):
        pT = 0
    elif (time< ( param_all['BT'] + param_all['ST']) or RUN == 'Ext'):
        pT = param_all['p'] + param_dict['psign']*(time-param_all['BT'])
    else:
        pT = 1-1.3*(param_dict['gamma']+param_dict['mu'])/param_dict['beta0']
        
    return pT

def AC_theory(results_dict, data_type, RUN, model_no,
                        param_all, param_dict, theory_func= None, odea_theory = False):
    statistic_dict = {}
    
    betaT = betaChange(param_dict=param_dict,
                       param_all=param_all,
                       RUN=RUN)
    if data_type == 'Incidence':
        ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
        theory = [np.nan for val in ts]
        theory_col = 'tab:green'
        theory_name = '(Poisson)'
        
    elif data_type == 'Prevalence':
        if model_no == 1:
            exponential_growth = -abs(betaT*(1-2*results_dict['P_theory'][:,0])-param_dict['gamma'])
            theory = np.exp(exponential_growth)
            ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
        elif model_no == 2:
            theory=autocorrelation_from_powerspectrum(results_dict=results_dict,
                                              param_all = param_all,
                                              param_dict = param_dict,
                                              RUN=RUN, 
                                              matrices_func=matrixVaccineModel,
                                              powerspectrum_func=PowerSpectrum_prevalence)
            ts = range(param_all['T']+param_all['BT'])
        elif model_no ==3:
            exponential_growth = -abs(betaT*(1-2*results_dict['P_theory'][:,0])-param_dict['gamma']-param_dict['nu'])
            theory = np.exp(exponential_growth)
            ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
        theory_col = 'tab:red'
        theory_name = '(SDE)'
    elif data_type == 'RoI':
        if model_no ==1:
            # exponential_growth = -abs(betaT*(1-2*results_dict['P_theory'][:,0])-param_dict['gamma']
            #                           - 2*(betaT*results_dict['P_theory'][:,0]*(1-results_dict['P_theory'][:,0])
            #                                - param_dict['gamma']*results_dict['P_theory'][:,0])/(1-2*results_dict['P_theory'][:,0]) )  
            # exponential_growth = -abs(betaT - param_dict['gamma'])
            exponential_growth = -abs(betaT*(1-2*results_dict['P_theory'][:,0])-param_dict['gamma'])
            theory = np.exp(exponential_growth)
            ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
        elif model_no ==2:
            theory=autocorrelation_from_powerspectrum(results_dict=results_dict,
                                              param_all = param_all,
                                              param_dict = param_dict,
                                              RUN=RUN, 
                                              matrices_func=matrixVaccineModel,
                                              powerspectrum_func=PowerSpectrum_RoI)
            ts = range(param_all['T']+param_all['BT'])
        elif model_no ==3:
            exponential_growth = -abs(betaT*(1-2*results_dict['P_theory'][:,0])-param_dict['gamma'] - param_dict['nu']
                                      - 2*betaT*(betaT*results_dict['P_theory'][:,0]*(1-results_dict['P_theory'][:,0])
                                           - param_dict['gamma']*results_dict['P_theory'][:,0]
                                           + param_dict['nu']*(1-results_dict['P_theory'][:,0]))/(betaT*(1-2*results_dict['P_theory'][:,0]) 
                                                                                                  - param_dict['nu']))  
            theory = np.exp(exponential_growth)
            ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
        theory_col = 'tab:orange'
        theory_name = '(SDE)'
    else:
        print('data-type name undefined')
    statistic_dict['theory'] = theory
    statistic_dict['theory_col'] = theory_col
    statistic_dict['ts'] = ts
    statistic_dict['theory_name'] = theory_name
    return statistic_dict


def Skew_theory(results_dict, data_type, RUN, model_no,
                        param_all, param_dict, theory_func= None, odea_theory = False):
    statistic_dict = {}
    ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
    if data_type == 'Incidence':
        theory = 1/np.sqrt(results_dict['I_theory'][:,0]*param_dict['N'])
        theory_col = 'tab:green'
        ts = ts
        theory_name = '(Poisson)'
       
    elif data_type == 'Prevalence':
        theory = np.zeros(len(ts))
        theory_col = 'tab:red'
        ts = ts
        theory_name = '(SDE)'
    elif data_type == 'RoI':
        theory = np.zeros(len(ts))
        theory_col = 'tab:orange'
        theory_name = '(SDE)'
    else:
        print('data-type name undefined')
    statistic_dict['theory'] = theory
    statistic_dict['theory_col'] = theory_col
    statistic_dict['ts'] = ts
    statistic_dict['theory_name'] = theory_name
    return statistic_dict

def CoV_theory(results_dict, data_type, RUN, model_no,
                        param_all, param_dict, theory_func= None, odea_theory = False):
    statistic_dict = {}
    ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
    if data_type == 'Incidence':
        theory = 1/np.sqrt(results_dict['I_theory'][:,0]*param_dict['N'])
        theory_col = 'tab:green'
        ts = ts
        theory_name = '(Poisson)'
        if odea_theory:
            theory_var = results_dict['I_theoryOdea'][1]
            theory_mean = results_dict['I_theoryOdea'][0]
            statistic_dict['odea_theory'] = np.sqrt(theory_var/param_dict['N'])/theory_mean
        
    elif data_type == 'Prevalence':
        if model_no == 2:
            theory=np.sqrt(results_dict['P_theory'][:,4]/param_dict['N'])/(results_dict['P_theory'][:,1])
        else:
            theory = np.sqrt(results_dict['P_theory'][:,1]/param_dict['N'])/(results_dict['P_theory'][:,0])
         
        theory_col = 'tab:red'
        ts = ts
        theory_name = '(SDE)'
    elif data_type == 'RoI':
        if model_no ==2:
            var_theory = theory_func(results_dict['P_theory'], param_dict['beta0'])
                                   
            theory = np.sqrt(var_theory[1]/param_dict['N'])/var_theory[0]
        else:
            phi_var_theory = odeint(theory_func, param_dict['I0'],ts, 
                 args = (RUN, param_all['BT'], param_dict['beta0'],
                          param_dict['gamma'],
                          param_dict['nu'],param_all['p'],
                          param_dict['psign'], ))
            mean_theory = NC(phi_var_theory, ts, param_all['BT'], 
                             0,param_all['T'], RUN, param_dict['beta0'], param_dict['gamma'],
                             param_all['p'],
                          param_dict['psign'])
            theory = np.sqrt(phi_var_theory[:,1]/param_dict['N'])/(mean_theory[:,0])
        theory_col = 'tab:orange'
        theory_name = '(SDE)'
    else:
        print('data-type name undefined')
    statistic_dict['theory'] = theory
    statistic_dict['theory_col'] = theory_col
    statistic_dict['ts'] = ts
    statistic_dict['theory_name'] = theory_name
    return statistic_dict

def variance_theory(results_dict, data_type, RUN, model_no,
                        param_all, param_dict, theory_func= None, odea_theory = False):
    statistic_dict = {}
    ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
    if data_type == 'Incidence':
        theory =results_dict['I_theory'][:,0]
        theory_col = 'tab:green'
        ts = ts
        theory_name = '(Poisson)'
        
        if odea_theory:
            theory_odea = results_dict['I_theoryOdea'][1]
            statistic_dict['odea_theory'] = theory_odea
        
    elif data_type == 'Prevalence':
        if model_no == 2:
            theory=results_dict['P_theory'][:,4]
        else:
            theory = results_dict['P_theory'][:,1]
        theory_col = 'tab:red'
        ts = ts
        theory_name = '(SDE)'
    elif data_type == 'RoI':
        if model_no ==2:
            theory = theory_func(results_dict['P_theory'], param_dict['beta0'])[1]
        else:
            theory = odeint(theory_func, param_dict['I0'],ts, 
                 args = (RUN, param_all['BT'], param_dict['beta0'],
                          param_dict['gamma'],
                          param_dict['nu'],param_all['p'],
                          param_dict['psign'], ))[:,1]
            
        theory_col = 'tab:orange'
        theory_name = '(SDE)'
    else:
        print('data-type name undefined')
    statistic_dict['theory'] = theory
    statistic_dict['theory_col'] = theory_col
    statistic_dict['ts'] = ts
    statistic_dict['theory_name'] = theory_name
    return statistic_dict