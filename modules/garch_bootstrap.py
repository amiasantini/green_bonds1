import numpy as np
import pandas as pd
import random
import os
from modules.more.math_functions import (flex_p, garch_flex_p,fit_marginals_flex_p,dcc_parameters, covariance_to_correlation, factor_analysis)

#INPUT PARAMETERS
n_ = 8  # number of indices
k_ = 2  # number of factors for factor analysis
tau_ = 120  #prior half life -> flexible prob, exponential decay
uncond_corr_dict={} #dictionary from stock pairs to periods and uncond_corr
tail_dep_dict={} 
r_t_dict={} #dictionary from stock pairs to periods and daily cond_corr
garch_dict={}
dcc_dict={}
mu_marg_dict={} #marginal distribution parameters
sigma2_marg_dict={}
eps_dict={} ##########new
sigma2_dict={} ##########new

    #SUBPERIODS
    #Dates in american date format
subperiods= {
  "all": ['10/15/2014','06/07/2021'],
  "bear": ['05/21/2015','04/11/2016'],
  "bull": ['12/01/2016','01/31/2018'],
  "prepandemic": ['01/03/2019','03/03/2020'],
  "pandemic": ['03/04/2020','06/07/2021']
}

#Load data
#upload index values
path = os.getcwd()
df_stocks=pd.read_excel(path + '\\input_data.xlsx', index_col=0) 
#create list of column names for graphs
stock_names = list(df_stocks.columns)
#Log returns
v_stock = np.array(df_stocks.iloc[:, :n_])
dx_all = np.diff(np.log(v_stock), axis=0)  #compounded return

for i_1 in[0,1]:  #position of the two green bond indices to use in the analysis
    for i in subperiods:
        t_first = subperiods[i][0]  #start date    
        t_last=subperiods[i][1]
        #select the returns of that subperiod
        dx=dx_all[df_stocks.index.get_loc(t_first):df_stocks.index.get_loc(t_last)+1,:]
                #+1 because the index is of the price date, but we are selecting the returns
        t_ = dx.shape[0]
                
        #Set flexible probabilities
        p = flex_p(t_, tau_)
        #the GARCH parameters will be fit via the maximum likelihood
        #with flexible probabilities approach
        
        #Fit a GARCH(1,1) on each time series of compounded returns
        param = np.zeros((4, n_)) #parameters a,b,c,mu
        sigma2 = np.zeros((t_, n_))
        epsilon = np.zeros((t_, n_)) #innovations
                
        for n in range(n_):
            param[:, n], sigma2[:, n], epsilon[:, n] = garch_flex_p(dx[:, n], p, rescale=True)
                
        garch_dict[i]=param #store values of the ith sub-period
        eps_dict[i]=epsilon ########new
        sigma2_dict[i]= sigma2 #######new

nu_df=pd.read_excel('excel_tables\\market_relationships_results.xlsx',sheet_name='Final_values',index_col=0,skiprows=[1,len(subperiods)+2])
nu_dict={"all":[[i for i in nu_df.iloc[0]], [i for i in nu_df.iloc[len(subperiods)]]],
          "bear":[[i for i in nu_df.iloc[1]], [i for i in nu_df.iloc[len(subperiods)+1]]],
          "bull":[[i for i in nu_df.iloc[2]], [i for i in nu_df.iloc[len(subperiods)+2]]],
          "prepandemic":[[i for i in nu_df.iloc[3]], [i for i in nu_df.iloc[len(subperiods)+3]]],
          "pandemic":[[i for i in nu_df.iloc[4]], [i for i in nu_df.iloc[len(subperiods)+4]]]}                                    




def garch_flex_p_bootstrap(dx,nreps,tau_):
    t=np.shape(dx)[0]
    n=np.shape(dx)[1]
    sim_logs=np.zeros((1,t,n)) #to store simulated log returns
    dx_new=np.zeros((t,1)) #to store simulated asset prices
    #generate new dataset
    for rep in range(nreps):
        dx_new=np.array([np.array(random.choices(dx[:,i], k=t)).T for i in range(n)]).T
        sim_logs=np.insert(sim_logs,rep,dx_new,axis=0)
    sim_logs=np.delete(sim_logs,nreps,0) #delete first row of returns: it is NaN
    #Set flexible probabilities
    p = flex_p(t, tau_)    
    #Fit a GARCH(1,1) MLPF on each time series of compounded returns
    #for each simulated dataset
    boot_param_garch=np.zeros((nreps,4,n))
    boot_sigma2=np.zeros((nreps,t,n))
    boot_epsilon=np.zeros((nreps,t,n))
    for rep in range(nreps): #loop over number of simulations
        #Prepare arrays to store values
        param_garch = np.zeros((4, n)) #parameters a,b,omega,mu
        sigma2 = np.zeros((t, n)) #conditional variances
        epsilon = np.zeros((t, n)) #innovations
        for i in range(n): #loop over number of assets
            param_garch[:, i], sigma2[:, i], epsilon[:, i] = garch_flex_p(sim_logs[rep,:, i], p, rescale=True)
        boot_param_garch[rep,:,:]=param_garch #store the values of all simulations
        boot_sigma2[rep,:,:]=sigma2
        boot_epsilon[rep,:,:]=epsilon
    return sim_logs, boot_param_garch, boot_sigma2, boot_epsilon


from scipy.stats import t
def dcc_function(epsilon,p,nu_dict,i_1,period,stock_names):
    n=np.shape(epsilon)[1]     
    dcc_period=[]
    #Estimate marginal distributions of the epsilon by fitting a Student t 
    #distribution via maximum likelihood with flexible probabilities
    #with known degrees of freedom nu (sufficiently high nu values lead to Gaussian case)
    mu_marg = np.zeros(n)
    sigma2_marg = np.zeros(n)
    for j in range(n):
        mu_marg[j], sigma2_marg[j] = fit_marginals_flex_p(epsilon[:, j], p=p, nu=nu_dict[period][i_1][j])
            
    #Map each marginal time series of epsilon into standardized realizations, xi
    xi = np.zeros((t_, n))
    for j in range(n):
        u = t.cdf(epsilon[:, j], df=nu_dict[period][i_1][j], loc=mu_marg[j],
                  scale=np.sqrt(sigma2_marg[j]))
        u[u <= 10**(-7)] = 10**(-7)
        u[u >= 1-10**(-7)] = 1-10**(-7)
        xi[:, j] = t.ppf(u, df=nu_dict[period][i_1][j]) 
    #we take the same dfs in marginals as in copula
               
    #LOOP OVER STOCKS (same i_1 and a different i_2)
    for i_2 in range(i_1+1,n_):
        
        #Estimate the unconditional correlation matrix via MLFP 
        #(sufficiently high nu values lead to Gaussian case)
        _, sigma2_xi = fit_marginals_flex_p(xi, p=p, nu=nu_dict[period][i_1][i_2])
        rho_xi = covariance_to_correlation(sigma2_xi)
        
        beta, delta2 = factor_analysis(rho_xi, k_)
        rho = covariance_to_correlation(beta @ beta.T + np.diag(delta2)) #factor analysis shrinkage
        #Fit DCC
        #the dcc function accounts for both gaussian and student t cases
        (c, a, b), r_t, epsi, q_t_ = dcc_parameters(xi, p, rho, nu_dict[period][i_1][i_2])
        dcc_period.append([stock_names[i_2],c,a,b])
    dcc_df=pd.DataFrame(dcc_period,columns=['Name','c','a','b'])
    return dcc_df #, r_t, epsi, q_t_
    

i='pandemic'
dcc_period=dcc_dict[i].iloc[:,1:].to_numpy()
     
def conf_interval_dcc(boot_epsilon,dcc_period,p,nu_dict,i_1,period,stock_names,tot_i1=2):    #outside loop over i_1 and period
    nreps=np.shape(boot_epsilon)[0]
    n=np.shape(boot_epsilon)[2]
    boot_dcc=np.zeros((nreps,n-tot_i1,3))
    #compute bootstrapped dcc for each sample epsilon
    for rep in range(nreps):
        epsilon=boot_epsilon[rep,:,:]
        dcc_df = dcc_function(epsilon,p,nu_dict,i_1,period,stock_names)
        boot_dcc[rep,:,:]=dcc_df.iloc[:,1:].to_numpy()
    indices=dcc_df['Name']
    #retrieve estimated parameter values from original sample
    dcc_0=dcc_period
    #find indices for confidence interval quantiles
    quantile_low=int(np.floor(0.024*nreps)) #for confidence interval
    quantile_up=int(np.floor(0.974*nreps)) #for confidence interval
    c_is=np.zeros((3,n-tot_i1,2))
    pvals=np.zeros((n-tot_i1,3))
    #sort simulated values for each parameter
    boot_dcc_sorted=np.sort(boot_dcc,axis=0) #every row is a simulation. columns are 8 indices
    #compute confidence intervals of c, a, b
    c_is[0,:,:]=[(boot_dcc_sorted[quantile_low,j,0],boot_dcc_sorted[quantile_up,j,0]) for j in range(n-tot_i1)]
    c_is[1,:,:]=[(boot_dcc_sorted[quantile_low,j,1],boot_dcc_sorted[quantile_up,j,1]) for j in range(n-tot_i1)]
    c_is[2,:,:]=[(boot_dcc_sorted[quantile_low,j,2],boot_dcc_sorted[quantile_up,j,2]) for j in range(n-tot_i1)]
    #compute pvalues of c, a, b
    pvals[:,0]=np.count_nonzero(boot_dcc[:,:,0]>dcc_0[:,0],axis=0)/nreps
    pvals[:,1]=np.count_nonzero(boot_dcc[:,:,1]>dcc_0[:,1],axis=0)/nreps
    pvals[:,2]=np.count_nonzero(boot_dcc[:,:,2]>dcc_0[:,2],axis=0)/nreps
    pvals_df=pd.DataFrame(pvals,index=indices,columns=['c','a','b'])
    return c_is, pvals_df


    
def conf_interval_garch_flex_p(dx,nsims,tau_,garch_period):
    quantile_low=int(np.floor(0.024*nsims)) #for confidence interval
    quantile_up=int(np.floor(0.974*nsims)) #for confidence interval
    n=np.shape(dx)[1]
    c_is=np.zeros((4,n,2))
    pvals=np.zeros((4,n))
    sim_logs, boot_param, _,_ = garch_flex_p_bootstrap(dx, nsims, tau_)
    #get estimated GARCH parameter values from original sample
    alpha_0=garch_period[0,:]
    beta_0=garch_period[1,:]
    omega_0=garch_period[2,:]
    mu_0=garch_period[3,:]
    #sort simulated values for each parameter
    alpha_sim=np.sort(boot_param[:,0,:],axis=0) #every row is a simulation. columns are 8 indices
    beta_sim=np.sort(boot_param[:,1,:],axis=0)
    omega_sim=np.sort(boot_param[:,2,:],axis=0)
    mu_sim=np.sort(boot_param[:,3,:],axis=0)
    #compute confidence intervals
    c_is[0,:,:]=[(alpha_sim[quantile_low,j],alpha_sim[quantile_up,j]) for j in range(n)]
    c_is[1,:,:]=[(beta_sim[quantile_low,j],beta_sim[quantile_up,j]) for j in range(n)]
    c_is[2,:,:]=[(omega_sim[quantile_low,j],omega_sim[quantile_up,j]) for j in range(n)]
    c_is[3,:,:]=[(mu_sim[quantile_low,j],mu_sim[quantile_up,j]) for j in range(n)]
    #compute pvalues
    pvals[0,:]=np.count_nonzero(alpha_sim>alpha_0,axis=0)/nsims
    pvals[1,:]=np.count_nonzero(beta_sim>beta_0,axis=0)/nsims 
    pvals[2,:]=np.count_nonzero(omega_sim>omega_0, axis=0)/nsims
    pvals[3,:]=np.count_nonzero(mu_sim>mu_0,axis=0)/nsims
    return c_is , pvals

























#Calling the functions for the bootstrap
nsims=10
ci_dict, pvals_dict = {},{} 
for i in subperiods:
    t_first = subperiods[i][0]    
    t_last=subperiods[i][1]
    dx=dx_all[df_stocks.index.get_loc(t_first):df_stocks.index.get_loc(t_last)+1,:]
    garch_period=garch_dict[i]
    ci_dict[i], pvals_dict[i] = conf_interval_garch_flex_p(dx,nsims,tau_,garch_period)

#Export Epsilon and Sigma2 for GARCH(1,1) fit test:
with pd.ExcelWriter("excel_tables/partialtables/garch_bootstrap.xlsx") as writer:
    for k in ci_dict:
        #confidence intervals dataframes
        df_ci_alpha=pd.DataFrame(ci_dict[k][0,:,:]) 
        df_ci_beta=pd.DataFrame(ci_dict[k][1,:,:]) 
        df_ci_omega=pd.DataFrame(ci_dict[k][2,:,:]) 
        df_ci_mu=pd.DataFrame(ci_dict[k][3,:,:]) 
        df_ci_alpha.index=stock_names
        df_ci_beta.index=stock_names
        df_ci_omega.index=stock_names
        df_ci_mu.index=stock_names
        #pvalues dataframe
        df_pvals=pd.DataFrame(pvals_dict[k])
        df_pvals.columns=stock_names
        df_pvals.index=['alpha','beta','omega','mu']
        #export to Excel
        df_ci_alpha.to_excel(writer, sheet_name='ci_alpha_%s' %k)
        df_ci_beta.to_excel(writer, sheet_name='ci_beta_%s' %k)
        df_ci_omega.to_excel(writer, sheet_name='ci_omega_%s' %k)
        df_ci_mu.to_excel(writer, sheet_name='ci_mu_%s' %k)
        df_pvals.to_excel(writer, sheet_name='pvals_%s' %k)
        

