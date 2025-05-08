import math 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

# Compute empirical quantiles from returns
def quantile(array,steps):
    array=np.array(array,ndmin=2)
    #array=array.transpose()
    if array.shape[0]==1:  ##
        array=array.transpose()  ##
    n_=array.shape[1]
    quantile_period=np.zeros((steps+1, n_))
    for q in range(0,steps+1,1):
        quantile_period[q]=np.quantile(array, q/steps, axis = 0) #interpolates linearly
    return quantile_period

# Define utility function for the Exponential (ERM) spectral measure
def utility(s,k):
    return k*math.exp(-k*s)/(1-math.exp(-k))

# Compute spectral risk measure
def spectral_measure(quantile_array,k,steps): 
    quantile_array=np.array(quantile_array,ndmin=2) #if quantile_function is not array
    if quantile_array.shape[0]==1:
        quantile_array=quantile_array.transpose()  #sometimes it gets transposed accidentally, this fixes it
    n_=quantile_array.shape[1]
    rho=np.zeros((1,n_))  #initialize rho, one place for each index
    for q in range(0,steps,1):
        rho+= 0.5*(utility(q/steps,k)*quantile_array[q]+utility((q+1)/steps,k)*quantile_array[q+1])*1/(steps)  #interval midpoint*length of interval
    return -1*rho

# Create the SRM plots
def plot_SRM(k_list,
             path, 
             folder,
             filename1='\excel_tables\partialtables\Spectral_output_GREEN.xlsx',
             filename2='\excel_tables\partialtables\Spectral_output_NO_GREEN.xlsx'):
    #import data
    spectral_dict_green_in={}
    spectral_dict_green_oos={}
    spectral_dict_nogreen_in={}
    spectral_dict_nogreen_oos={}
    for k in k_list:
        spectral_dict_green_in[k] = pd.read_excel(path + filename1,sheet_name="spectral_in %s" % k, index_col=0)
        spectral_dict_green_oos[k] = pd.read_excel(path + filename1,sheet_name="spectral_oos %s" % k, index_col=0)
        spectral_dict_nogreen_in[k] = pd.read_excel(path + filename2,sheet_name="spectral_in %s" % k, index_col=0)
        spectral_dict_nogreen_oos[k] = pd.read_excel(path + filename2,sheet_name="spectral_oos %s" % k, index_col=0)
    
    ns= spectral_dict_green_in[k_list[0]].shape[0] #number of strategies
    
    for k in k_list:
        # In-sample plots
        fig = plt.figure()
        fig, ax = plt.subplots(2, 2)
        fig.tight_layout()
        #fig.suptitle('In-sample spectral risk measure, k = %s' % k, fontweight="bold")
        plt.subplot(1, 2, 1)
        #plt.title('Green Portfolios')
        data=spectral_dict_green_in[k]
        markers=['p','*','d','+','x','^'] #as many as the value of ns
        labels=['Mean Var', 'Min Var', 'E. W.', 'R. P.', 'CVaR', 'Max Div']   #equal weights, risk parity
        for i in range(0,ns):
            plt.plot(data.transpose()[:][i], linewidth=0,marker=markers[i],label=labels[i])
        ax = plt.gca()
        ax.set_ylim([data.min().min()-0.01, data.max().max()+0.01])
        #ax.set_yscale('log')
        plt.yticks(fontsize=15) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xticks(fontsize=15, rotation=35) 
        ax.legend(fontsize=13)
        
        plt.subplot(1, 2, 2)
        #plt.title('Non-green Portfolios')
        data=spectral_dict_nogreen_in[k]
        markers=['p','*','d','+','x','^'] #as many as the value of ns
        labels=['Mean Var', 'Min Var', 'E. W.', 'R. P.', 'CVaR', 'Max Div']
        for i in range(0,ns):
            plt.plot(data.transpose()[:][i], linewidth=0,marker=markers[i],label=labels[i])
        ax = plt.gca()
        ax.set_ylim([data.min().min()-0.01, data.max().max()+0.01])
        plt.yticks(fontsize=15)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xticks(fontsize=15, rotation=35) 
        ax.legend(fontsize=13)
        
        plt.savefig('{}/{}-ins.png'.format(path+folder,k),bbox_inches='tight') 
        
        # Out-of-sample plots
        fig = plt.figure()
        fig, ax = plt.subplots(2, 2)
        fig.tight_layout()
        #fig.suptitle('Out-of-sample spectral risk measure, k = %s' % k, fontweight="bold")
        plt.subplot(1, 2, 1)
        #plt.title('Green Portfolios')
        data=spectral_dict_green_oos[k]
        markers=['p','*','d','+','x','^'] #as many as the value of ns
        labels=['Mean Var', 'Min Var', 'E. W.', 'R. P.', 'CVaR', 'Max Div']
        for i in range(0,ns):
            plt.plot(data.transpose()[:][i], linewidth=0,marker=markers[i],label=labels[i])
        ax = plt.gca()
        ax.set_ylim([data.min().min()-0.01, data.max().max()+0.01])
        plt.yticks(fontsize=15) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xticks(fontsize=15, rotation=35) 
        ax.legend(fontsize=13)
        
        plt.subplot(1, 2, 2)
        #plt.title('Non-green Portfolios')
        data=spectral_dict_nogreen_oos[k]
        markers=['p','*','d','+','x','^'] #as many as the value of ns
        labels=['Mean Var', 'Min Var', 'E. W.', 'R. P.', 'CVaR', 'Max Div']
        for i in range(0,ns):
            plt.plot(data.transpose()[:][i], linewidth=0,marker=markers[i],label=labels[i])
        ax = plt.gca()
        ax.set_ylim([data.min().min()-0.01, data.max().max()+0.01])
        plt.yticks(fontsize=15) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xticks(fontsize=15, rotation=35) 
        ax.legend(fontsize=13)
        plt.savefig('{}/{}-oos.png'.format(path+folder,k),bbox_inches='tight') 



