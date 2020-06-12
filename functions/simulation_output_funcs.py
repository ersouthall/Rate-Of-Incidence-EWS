import numpy as np
from scipy.integrate import odeint
import pandas as pd
from statsmodels.graphics.functional import banddepth


def PrevalenceTheoryM1(X, tt,RUN, BurnT,
                        beta0, gamma, p, nu=0,mu=0, p_0=0):
     '''
    ODE's for Model 1: SIS with decreasing beta model (elimination by social distancing)
    returns: prevalence mean-field equations (dI/dt), variance of fluctuations (Ivar, d\zeta/dt) and kurtosis of fluctuations (Ikurt)
    RUN: defines if beta(t) will be slowing decreasing for Ext simulations or fixed for null simulations 
    '''
    if (tt <BurnT or RUN=='Fix'):
        betaT = beta0
    elif (tt< ( BurnT) or RUN == 'Ext'):
        betaT = beta0*(1-p*(tt-BurnT))
    else:
        betaT = 1.3*gamma
    I = X[0]*(betaT*(1-X[0])-gamma)
    Ivar = 2*(betaT - gamma -2*betaT*X[0])*X[1]+ betaT*(1-X[0])*X[0]+gamma*X[0]
    Ikurt = 4*(betaT - gamma -2*betaT*X[0])*X[2] + 6*(betaT*(1-X[0])*X[0]+gamma*X[0])*X[1]
    return [I,Ivar, Ikurt]

def PrevalenceTheoryM2(X, tt,RUN, BurnT, beta0, gamma,p, nu,mu, p_0):
         '''
    ODE's for Model 2: SIS with increasing vaccination rate model (elimination by vaccination)
    returns: prevalence mean-field equations (dI/dt), variance of fluctuations (Ivar, d\zeta/dt) and kurtosis of fluctuations (Ikurt)
    RUN: defines if p(t) (rate of vaccine uptake) will be slowing increasing for Ext simulations or fixed for null simulations 
    '''
    if (tt <BurnT or RUN=='Fix'):
        pT = 0
    elif (tt< ( BurnT) or RUN == 'Ext'):
        pT = p_0 + p*(tt-BurnT)
    else:
        pT = 1-1.3*(gamma+mu)/beta0
    S = -beta0*X[0]*X[1]+mu*(1-pT-X[0])+gamma*X[1]
    I = X[1]*(beta0*X[0]-gamma-mu)
    A= np.matrix([[-beta0*X[1] - mu, -beta0*X[0]+gamma], [beta0*X[1], beta0*X[0] - mu - gamma]])
    B = np.matrix([[beta0*X[0]*X[1] + mu*(1-X[0]) +mu*pT +gamma*X[1], -X[1]*(beta0*X[0] + mu+gamma)],[-X[1]*(mu+gamma+beta0*X[0]), X[1]*(beta0*X[0] + mu+gamma)]])
    Θ = np.matrix([[X[2], X[3]],[X[3], X[4]]])

    K = A*Θ +Θ*A.transpose()+B
    return [S, I,K[0,0], K[0,1], K[1,1]]

def PrevalenceTheoryM3(X, tt,RUN, BurnT,
                        beta0, gamma,p, nu, mu=0, p_0=0):
    '''
    ODE's for Model 3: SIS emergence model with external infections
    returns: prevalence mean-field equations (dI/dt), variance of fluctuations (Ivar, d\zeta/dt) and kurtosis of fluctuations (Ikurt)
    RUN: defines if beta(t) will be slowing increasing for Em simulations or fixed for null simulations 
    '''
    if (tt <BurnT or RUN=='Fix'):
        betaT = beta0
    elif (tt< ( BurnT) or RUN == 'Ext'):
        betaT = beta0*(1+p*(tt-BurnT))
    else:
        betaT = 1.3*gamma
    I = betaT*X[0]*(1-X[0]) - gamma*(X[0]) + nu*(1-X[0])
    Ivar = 2*(betaT - gamma -2*betaT*X[0]-nu)*X[1]+ betaT*(1-X[0])*X[0]+gamma*X[0]+nu*(1-X[0])
    Ikurt = 4*(betaT - gamma -2*betaT*X[0]-nu)*X[2] +6*(betaT*(1-X[0])*X[0]+gamma*X[0]+nu*(1-X[0]))*X[1]
    return [I,Ivar, Ikurt]

def ODea(transmission, removal, importation, RUN, BT, ST, T, beta0, gamma, p, nu):
    '''
    O'Dea et al., 2019 equation 7 and 8
    Solution for the mean and variance of incidence data (from a SIS emergence system)
    
    '''
    betaFix = beta0*np.ones((BT+T)*10)
    betaExt =np.concatenate((beta0*np.ones(BT*10), beta0*(1+p*(np.arange(BT, BT+T, 0.1)-BT))))
    betaNExt = np.concatenate((betaExt[:(BT+ST)*10], 1.3*gamma*np.ones((T-ST)*10))) 
    if RUN =='Fix':
        betaT = betaFix
    if RUN == 'Ext':
        betaT = betaExt
    if RUN == 'NExt':
        betaT = betaNExt
    Mean = removal*importation/(removal - betaT)
    omega = (removal - betaT)/2
    SecondFact = 1+(betaT/(importation*omega))*(1-(1-np.exp(-2*omega))/(2*omega))
    var = (Mean**2)*(SecondFact-1)+Mean
    return [Mean, var]

def NCM2(phi, tt, BT, ST,T, RUN, beta0, gamma, p, sign):
    '''
    Theoretical solution for incidence for models with immunity (i.e when I != N - S), e.g. model 2
    In model 2, beta0 is fixed throughout (only rate of incidence changes which doesn't affect the theoretical solution)
    '''
    Results = np.zeros(((BT+T)*10, 2))
    Results[:,0] = beta0*phi[:,0]*phi[:,1]
    Results[:,1] = gamma*phi[:,1]
    
    return Results
def NC(phi, tt, BT, ST,T, RUN, beta0, gamma, p, sign):
    '''
    Theoretical solution for incidence for models where I = N -S 
    RUN: defines if beta(t) will be slowing increasing for Em/Ext simulations or fixed for null simulations 
    '''
    betaFix = beta0
    betaExt =np.concatenate((beta0*np.ones(BT*10), beta0*(1+sign*p*(np.arange(BT, BT+T, 0.1)-BT))))
    betaNExt = np.concatenate((betaExt[:(BT+ST)*10], 1.3*gamma*np.ones((T-ST)*10))) 
    if RUN =='Fix':
        betaT = betaFix
    if RUN == 'Ext':
        betaT = betaExt
    if RUN == 'NExt':
        betaT = betaNExt
    
    Results = np.zeros(((BT+T)*10, 2))
    Results[:,0] = betaT*(phi[:,0])*(1-phi[:,0])
    Results[:,1] = gamma*phi[:,0]
    return Results

def openFile(filePath, 
             realisations, T,BT,ST, 
             beta0, gamma, nu,mu,p0, p, Initial, sign,
             TYPE, 
             functionPrev, functionInc, functionOdea = None):
    Results = np.load(filePath,
                      allow_pickle=True)
    
    '''
    Function reads in simulation output and returns a dict of:
    I_sim - incidence simulations
    P_sim - prevalence simulations 
    I_theory - incidence theory
    P_theory - prevalence theory 
    I_theoryOdea - if available, O'Dea et al., solution 
    '''
    #realisations
    Inc = np.zeros(shape = (realisations, T+BT))
    Prev = np.zeros(shape = (realisations, (T+BT)*10))
    Sus = np.zeros(shape = (realisations, (T+BT)*10))
    for r in range(realisations):
        Inc[r,:] = Results[r][2]
        Prev[r,:] = Results[r][0]
        Sus[r, :] = Results[r][1]
#     #theory 
    ts = np.arange(0, T+BT, 0.1)
    
    PTheory = odeint(functionPrev, Initial, ts, args = (TYPE, BT, beta0, gamma,p,nu,mu, p0, ))
#     
    ITheory = functionInc(PTheory, ts,BT, ST, T, TYPE, beta0, gamma, p, sign)
    if functionOdea is not None:
        OdeaTheory = functionOdea(beta0, gamma, nu, TYPE,BT, ST, T, beta0, gamma,p, nu)
    else:
        OdeaTheory = 0
    
    return {'I_sim': Inc,'S_sim':Sus, 'P_sim': Prev, 'I_theory': ITheory, 'P_theory':PTheory, 'I_theoryOdea': OdeaTheory}

def RoI_fluctuationsM1M3(X, tt, RUN, BurnT, beta0, gamma,nu,p, psign):
    '''
    ODE's for the Rate of Incidence for models where I = N - S
    returns: prevalence mean-field (phi), rate of incidence variance (eta) and rate of incidence kurtosis (eta_kurt)
    '''
    if (tt <BurnT or RUN=='Fix'):
        betaT = beta0
    elif (tt< ( BurnT) or RUN == 'Ext'):
        betaT = beta0*(1+psign*p*(tt-BurnT))
    else:
        betaT = 1.3*gamma
    phi = betaT*X[0]*(1-X[0]) - gamma*X[0] + nu*(1-X[0])
    eta =(2*(betaT*(1-2*X[0])-gamma
               -nu
               -2*betaT*(betaT*X[0]*(1-X[0]) - gamma*X[0]+nu*(1-X[0]))/(betaT*(1-2*X[0])-nu))*X[1] +
            ((betaT*(1-2*X[0])-nu)**2)*(betaT*X[0]*(1-X[0]) + gamma*X[0]+nu*(1-X[0])) )
    eta_kurt = (4*(betaT*(1-2*X[0])-gamma
               -nu
               -2*betaT*(betaT*X[0]*(1-X[0]) - gamma*X[0]+nu*(1-X[0]))/(betaT*(1-2*X[0])-nu))*X[2] +
                6*(((betaT*(1-2*X[0])-nu)**2)*(betaT*X[0]*(1-X[0]) + gamma*X[0]+nu*(1-X[0])))*X[1])
    return [phi, eta, eta_kurt]

def RoI_fluctuationsM2(prevalence_theory_output, beta0):
    '''
    Approximation of the Rate of incidence for models where I!=N-S
    returns mean Rate of Incidence and variance rate of incidence 
    '''
    psi = prevalence_theory_output[:,0]
    phi = prevalence_theory_output[:,1]
    theta22 = prevalence_theory_output[:,4]
    theta11 = prevalence_theory_output[:,2]
    theta12 = prevalence_theory_output[:,3]
    
    mean_RoI = beta0*psi*phi
    variance_RoI = (beta0**2)*((psi**2)*theta22 + (phi**2)*theta11 + 2*phi*psi*theta12)
    return [mean_RoI, variance_RoI]

def RoI_approximation(outputs, TYPE, beta0, BT, T, ST,nu, gamma,p, sign,N,
       rolling_window = 20):
    '''
    Function which calculates Rate of Incidence from simulation output
    Method 1 (true RoI): relies on prevalence data and calculates rate of incidence as the product beta*S*I 
    Method 2 (rolling RoI): relies on incidence data and caluclates rate of incidence as the mean over a rolling window
    '''
    #method 1
    betaFix = beta0*np.ones((BT+T)*10)
    betaExt =np.concatenate((beta0*np.ones(BT*10), beta0*(1+sign*p*(np.arange(BT, BT+T, 0.1)-BT))))
    betaNExt = np.concatenate((betaExt[:(BT+ST)*10], 1.3*gamma*np.ones((T-ST)*10))) 
    if TYPE =='Fix':
        betaT = betaFix
    if TYPE == 'Ext':
        betaT = betaExt
    if TYPE == 'NExt':
        betaT = betaNExt
        
#     detrend_S = outputs['S_sim']-np.mean(outputs['S_sim'], 0)
#     detrend_I = outputs['P_sim']-np.mean(outputs['P_sim'], 0)
    detrend_S = outputs['S_sim']
    detrend_I = outputs['P_sim']
    lambda_1 = (betaT*detrend_S*detrend_I)/N + nu*detrend_S
    
    #method 2
    incidence = outputs['I_sim']
    df_incidence = pd.DataFrame(incidence.T)
    rolling_mean = df_incidence.rolling(window = rolling_window,
                                       center = True).mean().values.T
    rolling_var = df_incidence.rolling(window = rolling_window,
                                       center = True).var().values.T
    
    return lambda_1, rolling_mean, rolling_var, betaT

