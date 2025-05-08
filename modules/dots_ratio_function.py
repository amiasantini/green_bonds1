import numpy as np
import pandas as pd
import os
import win32com.client
from scipy.stats import t
from modules.checkpoint import checkpoint
from modules.more.math_functions import (flex_p, fit_marginals_flex_p, 
garch_flex_p, dcc_parameters, covariance_to_correlation, factor_analysis, spectral_dec_cov)

def dots_ratio_calibration(n_,k_,tau_,subperiods):
    #we set some potential values for the degrees of freedom parameter and look for the one
    #that will maximize the dots_ratio
    nu_calibration_list=[4.,5.,6.,7.,8.,10.,20.,30.,50.,100.,250.,500.,750.,1000.,1000000.]
    for i_1 in [0,1]:
        dots_ratio_dict={} #will hold dots_ratios by period
        for i in subperiods:   
            t_first = subperiods[i][0]  # starting date    
            t_last=subperiods[i][1]
            #LOAD DATA
            path = os.getcwd()
            df_stocks=pd.read_excel(path + '\input_data.xlsx', index_col=0)      
            # select data within the date range
            df_stocks = df_stocks.loc[(df_stocks.index >= t_first) &
                                      (df_stocks.index <= t_last)]      
            #create list of column names for graphs
            stock_names = list(df_stocks.columns)
            
            #COMPUTE LOG RETURNS
            v_stock = np.array(df_stocks.iloc[:, :n_])
            dx = np.diff(np.log(v_stock), axis=0)  #compounded return
            t_ = dx.shape[0]
           
            #SET FLEXIBLE PROBABILITIES
            p = flex_p(t_, tau_)  # flexible probabilities
            #the GARCH parameters will be fit via the maximum likelihood
            #with flexible probabilities approach
    
            #Fit a GARCH(1,1) on each time series of compounded returns
            param = np.zeros((4, n_)) #parameters a,b,c,mu
            sigma2 = np.zeros((t_, n_))
            epsilon = np.zeros((t_, n_)) #innovations      
            for n in range(n_):
                param[:, n], sigma2[:, n], epsilon[:, n] = \
                    garch_flex_p(dx[:, n], p, rescale=True)
            
            # uncond_corr_to_copy =[] #temporary list for uncond_corr in each period
            # tail_dep_array=np.zeros((4,n_-i_1-1))
            # r2_t_daily=np.zeros((t_,n_-i_1-1)) #temporary list for cond_corr in each period
            dots_ratio_period =np.zeros((len(nu_calibration_list),n_-1-i_1)) 
            #temporary array for dots_ratio in one period      
                
            #LOOP OVER STOCKS (same i_1 and a different i_2)
            for i_2 in range(i_1+1,n_):
                row=0
                for nu in nu_calibration_list:
                    #Estimate marginal distributions of epsilon by fitting a Student t 
                    #distribution via MLFP, with known degrees of freedom nu
                    mu_marg = np.zeros(n_)
                    sigma2_marg = np.zeros(n_)
                    for n in range(n_):
                        mu_marg[n], sigma2_marg[n] = fit_marginals_flex_p(epsilon[:, n], p=p, nu=nu)
                      
                    
                    #Map each marginal time series into standard t (not -normal-) realizations
                    xi = np.zeros((t_, n_))
                    for n in range(n_):
                        u = t.cdf(epsilon[:, n], df=nu, loc=mu_marg[n], #!!!
                                  scale=np.sqrt(sigma2_marg[n]))
                        u[u <= 10**(-7)] = 10**(-7)
                        u[u >= 1-10**(-7)] = 1-10**(-7)
                        xi[:, n] = t.ppf(u, df=nu)          
                    
                    
                    _, sigma2_xi = fit_marginals_flex_p(xi, p=p, nu=nu)  #!!!
                    rho2_xi = covariance_to_correlation(sigma2_xi)
                    
                    beta, delta2 = factor_analysis(rho2_xi, k_)
                    rho2 = covariance_to_correlation(beta @ beta.T + np.diag(delta2))
                   #to parametrize nu
                             
                    
                    #DCC fit
                    (c, a, b), r2_t, epsi, q2_t_ = dcc_parameters(xi, p, rho2=rho2, nu=nu)
                    
                    q2_t_nextstep = c*rho2 +\
                                    b*q2_t_ +\
                                    a*(np.array([epsi[-1, :]]).T@np.array([epsi[-1, :]]))
                    r2_t_nextstep = covariance_to_correlation(q2_t_nextstep)
                    
        
                    # Count Ellipsoid points
                    xi_plot = epsilon[:, [i_1, i_2]]  #epsilon values of the pair under consideration
                    # Unconditional ellipsoid
                    s2_unc=rho2[np.ix_([i_1, i_2], [i_1, i_2])]
                    #spectral decomposition of covariance matrix in eigenvectors and eigenvalues
                    e_unc, lambda2_unc = spectral_dec_cov(s2_unc) #e are the eigenvectors
                    Diag_lambda_unc = np.diagflat(np.sqrt(np.maximum(lambda2_unc, 0))) #sqrt(eigenvalues)
                    # count points inside and outside, ratio of the two
                    p_inside_unc= sum(1 for value in xi_plot if checkpoint(value[:][0],value[:][1], e_unc, Diag_lambda_unc)==True)
                    p_outside_unc =sum(1 for value in xi_plot if checkpoint(value[:][0],value[:][1], e_unc, Diag_lambda_unc)==False)
                    ratio_unc = p_inside_unc/p_outside_unc
    
                    #Save dots_ratio per nu, per i_2
                    dots_ratio_period[row,i_2-2]=ratio_unc
                    row+=1 #for dots_ratio storage
                            
            #Store dot ratio by period
            dots_ratio_df=pd.DataFrame(dots_ratio_period, columns=stock_names[i_1+1:])
            dots_ratio_dict[i]=dots_ratio_df
            
        #Export dots ratio
        index_list=[int(i) for i in nu_calibration_list]
        if i_1==0:
            with pd.ExcelWriter("excel_tables/partialtables/dots_ratio_0.xlsx") as writer:
                for k in dots_ratio_dict:
                    dots_ratio_dict[k].index= index_list
                    pd.DataFrame(dots_ratio_dict[k]).to_excel(writer, sheet_name="%s" % k)
        elif i_1==1:
            with pd.ExcelWriter("excel_tables/partialtables/dots_ratio_1.xlsx") as writer:
                for k in dots_ratio_dict:
                    dots_ratio_dict[k].index= index_list
                    pd.DataFrame(dots_ratio_dict[k]).to_excel(writer, sheet_name="%s" % k)
        
    #refresh the excel file which finds the optimized degrees of freedom
    #it will be used to import the correct degrees of freedom when calibration_flag=False
    xlapp = win32com.client.DispatchEx("Excel.Application")
    wb = xlapp.Workbooks.Open(path + "/excel_tables/market_relationships_results.xlsx",UpdateLinks=True)
    wb.RefreshAll()
    xlapp.CalculateUntilAsyncQueriesDone()
    wb.Save()
    xlapp.Quit()
    return


def copula_calibration():
    
    return




