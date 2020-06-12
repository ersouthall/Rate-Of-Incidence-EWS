import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from output_funcs import RoI_approximation
from stats_funcs import AC_simulations
import numpy as np
import pandas as pd 
import seaborn as sns

#################################################
# Code to make figures and supp. figures in paper
# statistical_func_theory: insert function name from stats_funcs.py (theory)
# statistical_func_simulation: insert corresponding function name from stats_funcs.py (simulation)
# theory_func: function for RoI variance (either RoI_fluctuationsM2 or RoI_fluctuationsM1M3)
# r0_1_t: time point when R0 = 1
def IncPrev_figures(results_dict, param_dict,model_no,data_type,
                              ax, param_all, 
                              statistical_func_theory,statistical_func_simulation,
                               RUN='Ext', r0_1_t = 700, theory_func = None,
                              border_col = '#e0ecf4', text_col ='k', alpha =1, linestyle = '-',
                              odea_theory = False, fig_main_text = False):
                                                                    
    # Figure Border for all 
    findAxes = make_axes_locatable(ax)
    if (model_no ==1) or (fig_main_text) :
        
        
        topBorder = findAxes.append_axes("top", size="18%", pad=0)
        topBorder.get_xaxis().set_visible(False)
        topBorder.get_yaxis().set_visible(False)
        topBorder.set_facecolor(border_col) 
        topText = AnchoredText(data_type,loc=10, pad =0,
                      prop=dict(backgroundcolor=border_col,
                                size=16,
                                color=text_col,
                                rotation = 0, 
                               fontweight="bold"))
        topBorder.add_artist(topText)
    # Data
    ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)
    t_aggregated = range(param_all['T']+param_all['BT'])
    
    param_model = dict(param_all, **param_dict)
    statistic_simulation = statistical_func_simulation(results_dict, data_type, param_model)
    statisitc_theory = statistical_func_theory(results_dict, data_type, RUN, model_no,
                        param_all, param_dict)
    
    if (data_type == 'Prevalence') or (fig_main_text):
        # Figure right border only for prevalence or if in main text (e.g. Fig. 2)
        rightBorder = findAxes.append_axes("right", size="18%", pad=0)
        rightBorder.get_xaxis().set_visible(False)
        rightBorder.get_yaxis().set_visible(False)
        rightBorder.set_facecolor(border_col)
        if model_no==1:
            name = 'Elimination \n(social distancing)'
        elif model_no==2:
            name = 'Elimination \n(vaccination)'
        elif model_no ==3:
            name = 'Emergence \n'
        rightText = AnchoredText(name,
                                 loc=10,
                                 borderpad = 1,
                                 frameon=False,
                          prop=dict(backgroundcolor=border_col,
                                    size=16,
                                    color=text_col,
                                    rotation = 270, 
                                   fontweight = "bold"))
        rightBorder.add_artist(rightText)
    if data_type == 'Prevalence':
        t_aggregated = ts
    if statistical_func_simulation == AC_simulations:
        ax.plot(range(param_model['windowSize']//2, param_all['T']+param_all['BT']-param_model['windowSize']//2), 
                statistic_simulation[param_model['windowSize']//2:(param_all['T']+param_all['BT']-param_model['windowSize']//2)],
                linestyle, c='tab:blue',alpha =alpha, label = 'Gillespie Simulations'  )
        
        
    else:
        ax.plot(t_aggregated, statistic_simulation,linestyle, c='tab:blue',alpha =alpha, label = 'Gillespie Simulations'  )
        

    ax.plot(statisitc_theory['ts'], statisitc_theory['theory'],linestyle,
            lw = '2',
            c=statisitc_theory['theory_col'],
            alpha = alpha,
            label = 'Analytical '+statisitc_theory['theory_name'] )
    
    if odea_theory:
        theory_odea = statistical_func_theory(results_dict, data_type, RUN, model_no,
                        param_all, param_dict, odea_theory=odea_theory)['odea_theory']
        
        ax.plot(ts[:int(r0_1_t*10//1)], theory_odea[:int(r0_1_t*10//1)], linestyle,
                lw = '2', 
                c='#af8dc3',
                alpha = alpha, 
                label = "Analytical (O'Dea et al.)")

    ax.plot([r0_1_t, r0_1_t], [-1000000, 1000000],'--',lw=1, c='k',alpha =alpha, label =r'$R_0=1$')
    
    
def RoI_figures(results_dict, param_dict,model_no,data_type,
                 theory_func,RUN, ax, param_all,
                statistical_func_theory,statistical_func_simulation,
                    r0_1_t = 700, border_col = '#e0ecf4', text_col ='k', alpha =1, linestyle = '-',odea_theory = False):
    # Theory 
    statisitc_theory = statistical_func_theory(results_dict, 'RoI', RUN, model_no,
                        param_all, param_dict, theory_func=theory_func)
    
    ts = np.arange(0, param_all['T']+param_all['BT'], 0.1)

    roi_true, roi_approx_mean, roi_approx_var, betaT = RoI_approximation(results_dict,
                                                                         RUN,
                                                                         param_dict['beta0'],
                                                                         param_all['BT'],
                                                                         param_all['T'],
                                                                         param_all['ST'],
                                                                         param_dict['nu'],
                                                                         param_dict['gamma'],
                                                                         param_all['p'],
                                                                         sign = param_dict['psign'],
                                                                         N = param_dict['N'],
                                                                         rolling_window=param_dict['BW']
                                                                        )
    
    param_model = dict(param_all, **param_dict)
    simulation_true = statistical_func_simulation(roi_true, 'RoI', param_model)                                                                    
    simulation_approx = statistical_func_simulation(roi_approx_mean, 'RoI', param_model)                                                                    

    # Plots
    if statistical_func_simulation == AC_simulations:
        ax.plot(range(param_model['windowSize']//2, param_all['T']+param_all['BT']-param_model['windowSize']//2), 
                simulation_true[param_model['windowSize']//2:(param_all['T']+param_all['BT']-param_model['windowSize']//2)],
                linestyle, c='purple',alpha=alpha, label = 'True RoI')
        
        ax.plot(range(param_model['windowSize']//2, param_all['T']+param_all['BT']-param_model['windowSize']//2), 
                simulation_approx[param_model['windowSize']//2:(param_all['T']+param_all['BT']-param_model['windowSize']//2)],
                linestyle,c='tab:blue', alpha = alpha, label = 'Rolling RoI')
        
                
    else:
        ax.plot(ts, simulation_true,linestyle, c='purple',alpha=alpha, label = 'True RoI')

        ax.plot(simulation_approx,linestyle,c='tab:blue', alpha = alpha, label = 'Rolling RoI')
    ax.plot(statisitc_theory['ts'],statisitc_theory['theory'], linestyle,c=statisitc_theory['theory_col'],
            lw =2,alpha = alpha, label = 'Analytical '+statisitc_theory['theory_name'])
    ax.plot([r0_1_t, r0_1_t], [- 1000000, 1000000],'--',lw=1, c='k', alpha=alpha, label =r'$R_0=1$')

    findAxes = make_axes_locatable(ax)
    if model_no == 1:
        topBorder = findAxes.append_axes("top", size="18%", pad=0)
        topBorder.get_xaxis().set_visible(False)
        topBorder.get_yaxis().set_visible(False)
        topBorder.set_facecolor(border_col) 
        topText = AnchoredText("Rate of Incidence",loc=10, pad =0,
                      prop=dict(backgroundcolor=border_col,
                                size=16,
                                color=text_col,
                                rotation = 0, 
                                fontweight = "bold"))
        topBorder.add_artist(topText)
    if data_type == 'Figure 2':
        # right border on all ax if Figure 2
        rightBorder = findAxes.append_axes("right", size="18%", pad=0)
        rightBorder.get_xaxis().set_visible(False)
        rightBorder.get_yaxis().set_visible(False)
        rightBorder.set_facecolor(border_col)
        if model_no==1:
            name = 'Elimination \n(social distancing)'
        elif model_no==2:
            name = 'Elimination \n(vaccination)'
        elif model_no ==3:
            name = 'Emergence \n'
        rightText = AnchoredText(name,
                           loc=10,
                           borderpad = 1,
                           frameon=False,
                          prop=dict(backgroundcolor=border_col,
                                    size=16,
                                    color=text_col,
                                    rotation = 270,
                                    fontweight = "bold"))

        rightBorder.add_artist(rightText)
        if model_no != 1:
            topBorder = findAxes.append_axes("top", size="18%", pad=0)
            topBorder.get_xaxis().set_visible(False)
            topBorder.get_yaxis().set_visible(False)
            topBorder.set_facecolor(border_col) 
            topText = AnchoredText("Rate of Incidence",loc=10, pad =0,
                          prop=dict(backgroundcolor=border_col,
                                    size=16,
                                    color=text_col,
                                    rotation = 0, 
                                    fontweight = "bold"))
            topBorder.add_artist(topText)

def AUC_figure3(model, files, ax):
    df = createPandasDF(model, files)
    
    bars = sns.barplot(x="variable",y='value',
            hue="Type", data=df,edgecolor="1",
           ax=ax)
    line_auc_random = plt.hlines(y=0.5, xmin=-0.5, xmax=5-0.4,
                   colors='k', linestyles='dashed',label='Random (AUC=0.5)')
    bars.legend_.remove()
    
    for i, bar in enumerate(ax.patches):
        if i<5:
            hatch='x'
            col = 'tab:orange'
        elif 5<=i<10:
            hatch = 'x'
            col = 'tab:green'
        elif 10<= i <15:
            hatch = 'x'
            col = 'tab:red'
        elif 15<=i<20:

            hatch=''
            col = 'tab:orange'
        elif 20<=i<25:
            hatch = ''
            col = 'tab:green'
        else:
            hatch=''
            col ='tab:red'
        bar.set_hatch(hatch)
        bar.set_facecolor(col)
        bar.set_alpha(0.7)
    # Legend Information 
    RoI_col = mpatches.Patch( facecolor='tab:orange',
                       alpha=0.8,
                       hatch=r'',
                       linewidth = 1.3,
                       label='Rate of Incidence')

    Incidence_col = mpatches.Patch( facecolor='tab:green',
                           alpha=0.8,
                           hatch=r'',
                           linewidth = 1.3,
                           label='Incidence')
    Prevalence_col= mpatches.Patch( facecolor='tab:red',
                          alpha=0.8,hatch='',label='Prevalence')
    time_1_hatch = mpatches.Patch(facecolor='#bdbdbd',edgecolor= 'white',alpha=1,hatch='x',label=r'$t_1=390$')
    time_2_hatch = mpatches.Patch(facecolor='#bdbdbd',edgecolor= 'white',alpha=1,hatch='',label=r'$t_2=450$')
    
    
    return [RoI_col, Incidence_col, Prevalence_col, time_1_hatch, time_2_hatch, line_auc_random] 

def createPandasDF(model, filenames):
    '''model inputs: "Model1", "Model2" or "Model3"'''
    auc_results = [file for file in filenames if model in file] 
    
    endTimes = ['390', '450']
    aucTimes = {}
    for t in endTimes:
        aucFile = [file for file in auc_results if t in file][0]
        dfScores = pd.read_csv(aucFile, 
                              names = ['Variance',
                                         'Coef. of V',
                                         'Skewness',
                                         'Kurtosis',
                                         'AC(1)'])
        dfScores['Type'] = ['RoI'+t, 'I'+t, 'P'+t]
        dfScores = pd.melt(dfScores, id_vars = ['Type'])
        dfScores['Time'] = t
        aucTimes[t] = dfScores
        
    df = pd.concat(aucTimes, ignore_index=True)
    return df