import numpy as np
import statsmodels
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import t
from scipy.stats import rankdata
from modules.empirical_copula import tail
from modules.more.LiMakTest import LiMakTest
from modules.dots_ratio_function import dots_ratio_calibration
from modules.more.math_functions import (flex_p, fit_marginals_flex_p, garch_t_flex_p, 
                                         dcc_parameters, dcc_parameters_old, covariance_to_correlation, 
                                         factor_analysis, exp_cov_ellipsoid, 
                                         conf_interval_garch_flex_p, 
                                         conf_interval_dcc, mean_and_cov)
#increase plot image resolution:
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

#INPUT PARAMETERS
n_ = 8  # number of indices
k_ = 2  # number of factors for factor analysis
tau_ = 120  #prior half life -> flexible prob, exponential decay

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

#Engle's (1982) ARCH-LM test for Autoregressive Conditional Heteroskedasticity 
engle_p=np.zeros((1, dx_all.shape[1]))
for i in range(dx_all.shape[1]):
    engle_p[0,i]=statsmodels.stats.diagnostic.het_arch(dx_all[:,i], nlags=1)[1]
if engle_p.all()<0.01:
    print("Engle's (1982) ARCH-LM test for Autoregressive Conditional Heteroskedasticity:")
    print("The time series of log returns exhibit no heteroskedasticity (99% significance level).")
else:
    print("Engle's (1982) ARCH-LM test for Autoregressive Conditional Heteroskedasticity:")
    print("The time series of log returns exhibit heteroskedasticity (99% significance level).")
print("-------------------------------------")

#Start anaysis
for calibration_flag in [False]: #True if initialization step (all dfs=4)
    for i_1 in[0,1]:  #position of the two green bond indices to use in the analysis
        #We state the optimal degrees of freedom:
        if calibration_flag==True:
            #Dictionary TO START CALBRATION 
            nu_dict={"bear":[[4.,4.,4.,4.,4.,4.,4.,4.], [4.,4.,4.,4.,4.,4.,4.,4.]],
                      "bull":[[4.,4.,4.,4.,4.,4.,4.,4.], [4.,4.,4.,4.,4.,4.,4.,4.]],
                      "prepandemic":[[4.,4.,4.,4.,4.,4.,4.,4.], [4.,4.,4.,4.,4.,4.,4.,4.]],
                      "pandemic":[[4.,4.,4.,4.,4.,4.,4.,4.], [4.,4.,4.,4.,4.,4.,4.,4.]],
                      "all":[[4.,4.,4.,4.,4.,4.,4.,4.], [4.,4.,4.,4.,4.,4.,4.,4.]]}                                    
        else:
            #The correct one (after calibration)
            nu_df=pd.read_excel('excel_tables\\market_relationships_results.xlsx',sheet_name='Final_values',index_col=0,skiprows=[1,len(subperiods)+2])
            nu_dict={"all":[[i for i in nu_df.iloc[0]], [i for i in nu_df.iloc[len(subperiods)]]],
                      "bear":[[i for i in nu_df.iloc[1]], [i for i in nu_df.iloc[len(subperiods)+1]]],
                      "bull":[[i for i in nu_df.iloc[2]], [i for i in nu_df.iloc[len(subperiods)+2]]],
                      "prepandemic":[[i for i in nu_df.iloc[3]], [i for i in nu_df.iloc[len(subperiods)+3]]],
                      "pandemic":[[i for i in nu_df.iloc[4]], [i for i in nu_df.iloc[len(subperiods)+4]]]}                                    
                                          
        #Degrees of freedom (dfs) must be at least 4 for covariance values and
        #mahalanobis distance to work within optimization procedure.
        
        #We create dictionaries to store parameter values for table
        uncond_corr_dict={} #dictionary from stock pairs to periods and uncond_corr
        tail_dep_dict={} 
        r_t_dict={} #dictionary from stock pairs to periods and daily cond_corr
        garch_dict={}
        dcc_dict={}
        mu_marg_dict={} #marginal distribution parameters
        sigma2_marg_dict={}
        eps_dict={}
        sigma2_dict={}
        marginal_dfs_dict={}
        copula_dfs_dict={}

        for i in subperiods:
            t_first = subperiods[i][0]  #start date    
            t_last=subperiods[i][1]
            
            #Load data
            #upload index values
            path = os.getcwd()
            df_stocks=pd.read_excel(path + "\\input_data.xlsx", index_col=0) 
            
            #select data within the sub-period date range
            df_stocks = df_stocks.loc[(df_stocks.index >= t_first) &
                                      (df_stocks.index <= t_last)]
                    
            #create list of column names for graphs
            stock_names = list(df_stocks.columns)
            
            #Log returns
            v_stock = np.array(df_stocks.iloc[:, :n_])
            dx = np.diff(np.log(v_stock), axis=0)  #compounded return
            t_ = dx.shape[0]
        
                   
            #Set flexible probabilities
            p = flex_p(t_, tau_)
            #the GARCH parameters will be fit via the maximum likelihood
            #with flexible probabilities approach
            
            #Fit a GARCH(1,1) on each time series of compounded returns
            param = np.zeros((4, n_)) #parameters a,b,c,mu
            sigma2 = np.zeros((t_, n_))
            epsilon = np.zeros((t_, n_)) #innovations
            marginal_dfs_period=np.zeros(n_)
            for n in range(n_):
                param[:, n], sigma2[:, n], epsilon[:, n], marginal_dfs_period[n] = garch_t_flex_p(dx[:, n], p, rescale=True)
                #param[:, n], sigma2[:, n], epsilon[:, n], _ = garch_t_flex_p(dx[:, n], p, dfs= marginal_dfs_period[n], rescale=True)
            garch_dict[i]=param #store values of the ith sub-period
            eps_dict[i]=epsilon ########new
            sigma2_dict[i]= sigma2 #######new
            marginal_dfs_dict[i]= marginal_dfs_period #####new
            
            #Estimate marginal distributions of the epsilon by fitting a Student t 
            #distribution via maximum likelihood with flexible probabilities
            #with known degrees of freedom nu (sufficiently high nu values lead to Gaussian case)
            mu_marg = np.zeros(n_)
            sigma2_marg = np.zeros(n_)
            for n in range(n_):
                #mu_marg[n], sigma2_marg[n] = fit_marginals_flex_p(epsilon[:, n], p=p, nu=nu_dict[i][i_1][n])
                mu_marg[n], sigma2_marg[n] = fit_marginals_flex_p(epsilon[:, n], p=p, nu=marginal_dfs_dict[i][n])
              
            mu_marg_dict[i]=mu_marg  #add them to the dictionary, assigned to the corresponding sub-period
            sigma2_marg_dict[i]=sigma2_marg
            
            #Map each marginal time series of epsilon into standardized realizations, xi
            xi = np.zeros((t_, n_))
            for n in range(n_):
                u = t.cdf(epsilon[:, n], df=marginal_dfs_dict[i][n], loc=mu_marg[n],
                          scale=np.sqrt(sigma2_marg[n]))
                u[u <= 10**(-7)] = 10**(-7)
                u[u >= 1-10**(-7)] = 1-10**(-7)
                xi[:, n] = t.ppf(u, df=marginal_dfs_dict[i][n]) 

            
            uncond_corr_to_copy =[] #temporary list for uncond_corr in each period
            tail_dep_array=np.zeros((4,n_-i_1-1))
            r_t_daily=np.zeros((t_,n_-i_1-1)) #temporary list for cond_corr in each period
            dcc_period=[]
            
            #Estimate the unconditional correlation matrix via MLFP
            #we create a list for the period's nu, corresponding to the selected i_1
            copula_dfs_period=np.zeros(n_)

                
            #LOOP OVER STOCKS (same i_1 and a different i_2)
            for i_2 in range(i_1+1,n_):
                
                #Estimate the unconditional correlation matrix via MLFP 
                #(sufficiently high nu values lead to Gaussian case)
                _, sigma2_xi = mean_and_cov(xi, p)
                #_, sigma2_xi = fit_marginals_flex_p(xi, p=p, nu=nu_dict[i][i_1][i_2])
                rho_xi = covariance_to_correlation(sigma2_xi)
                
                beta, delta2 = factor_analysis(rho_xi, k_)
                rho = covariance_to_correlation(beta @ beta.T + np.diag(delta2)) #factor analysis shrinkage
                
                #Fit DCC
                #the dcc function accounts for gaussian and student t cases
                
                (c, a, b), r_t, epsi, q_t_, nu = dcc_parameters(xi[:,[i_1,i_2]], p, rho[[i_1,i_2],:][:,[i_1,i_2]])
                copula_dfs_period[i_2]=nu
                 
                
                q_t_next = c*rho[[i_1,i_2],:][:,[i_1,i_2]] +\
                                b*q_t_ +\
                                a*(np.array([epsi[-1, :]]).T@np.array([epsi[-1, :]]))
                r_t_next = covariance_to_correlation(q_t_next)
        

                if calibration_flag==False:  #we plot the ellipsoids once the unconditional correlation
                                            #value is based on the correct dfs
                    #Scatter plot of the epsilon            
                    epsilon_plot = epsilon[:, [i_1, i_2]]
                    fig = plt.figure()
                    plt.scatter(epsilon[:, i_1], epsilon[:, i_2], 2, marker='o', linewidths=1)
                    plt.axis('equal')
                    plt.axis([np.percentile(epsilon_plot[:, 0], 2), np.percentile(epsilon_plot[:, 0], 98),
                              np.percentile(epsilon_plot[:, 1], 2), np.percentile(epsilon_plot[:, 1], 98)])
                    plt.xlabel(r'$\epsilon_{%s}$' % stock_names[i_1], fontsize = 18)
                    plt.ylabel(r'$\epsilon_{%s}$' % stock_names[i_2], fontsize = 18)
                    plt.ticklabel_format(style='sci', scilimits=(0, 0))
                    #Overlaid ellipsoids
                    mu_plot = np.zeros(2)
                    rho_plot = rho[np.ix_([i_1, i_2], [i_1, i_2])]
                    #rho_plot = rho
                    r_t_plot = r_t_next
                    #r_t_plot = r_t_next
                    ell_unc = exp_cov_ellipsoid(mu_plot, rho_plot, color='b')
                    #ell_cond = exp_cov_ellipsoid(mu_plot, r_t_plot, color='tomato')

                    plt.legend([r'Innovations: $\epsilon_{i,j}$',r'Unconditional correlation: $\overline{q}_{i,j}$=%1.2f %%' %
                                 (100*rho_plot[0, 1])], fontsize = 15)
                    
                    if i_2==1:  #avoid plotting twice i_1 vs 1_2 and i_1 vs i_2
                        pass
                    else:
                        nr=i_2-1
                        if i_1==0:
                            fig.savefig('plots/ellipsoids/e{}{}{}.png'.format(i,'a',nr),bbox_inches='tight')
                        else:
                            fig.savefig('plots/ellipsoids/e{}{}.png'.format(i,nr),bbox_inches='tight')
        
                if calibration_flag==True:
                    #COPULA FITTING (initialization step)
                    #Find data ranks of variables i1 and i2
                    epsilon_plot = epsilon[:, [i_1, i_2]]
                    rk1 = rankdata(epsilon_plot[:,0],method='ordinal')
                    rk2 = rankdata(epsilon_plot[:,1],method='ordinal')
                    #obtain empirical univariate cdfs
                    length=len(rk1)
                    u1=rk1/length #standardized ranks
                    u2=rk2/length # ! len(rk1)=len(rk2)
                    
                    #EMPIRICAL TAIL DEPENDENCE
                    k=0.05 #percentile for tail dependence (lower)
                    tail_u=tail(u1,u2,1-k,1,1-k,1)/(k*length)
                    tail_l=tail(u1,u2,0,k,0,k)/(k*length)
                    
                    tail_u_inv=tail(u1,u2,0,k,1-k,1)/(k*length)
                    tail_l_inv=tail(u1,u2,1-k,1,0,k)/(k*length)
                    
                    # #Empirical Distribution Function 
                    # fig = plt.figure()
                    # plt.scatter(u1,u2,2, marker='o', linewidths=1)
                    # plt.xlabel('$U_{%s}$' % stock_names[i_1])
                    # plt.ylabel('$U_{%s}$' % stock_names[i_2])
                    # plt.title('Empirical uniform marginals scatterplot')
                    # plt.suptitle('Period: %s' % i.title())
                    # plt.show()
                    
                    #Save tail dependence coefficients per stock (single period)
                    tail_dep_array[0,i_2-i_1-1]= tail_u
                    tail_dep_array[1,i_2-i_1-1]= tail_u_inv
                    tail_dep_array[2,i_2-i_1-1]= tail_l
                    tail_dep_array[3,i_2-i_1-1]= tail_l_inv
                
                #Save unconditional correlation per stock (single period)
                uncond_corr_to_copy.append([rho[np.ix_([i_1, i_2], [i_1, i_2])][0,1],stock_names[i_2]])

                #Save daily r_t of each stock to database (single period)
                for day in range(0,t_):
                    r_t_daily[day,i_2-i_1-1]=r_t[day,0,1]
                    
                #Save i_2 DCC parameters (single period)
                dcc_period.append([stock_names[i_2],c,a,b])
            
            #Store tail dep data for each period
            #only for the initial calibration stage, as it does not change with the dfs
            if calibration_flag==True:
                tail_dep_dict[i]=tail_dep_array
            
            #Store unconditional correlation by period
            uncond_corr_df=pd.DataFrame(uncond_corr_to_copy,columns=['Unconditional Correlation','Name'])
            uncond_corr_dict[i]=uncond_corr_df
            
            #Store dates for plot of r_t
            if i=='all':
                dates= df_stocks.index
                dates=dates[:len(dates)-1]
            
            #Store conditional correlation by period
            r_t_df=pd.DataFrame(r_t_daily,columns=stock_names[i_1+1:n_])
            r_t_dict[i]=r_t_df
        
            #Store DCC parameters by period
            dcc_df=pd.DataFrame(dcc_period,columns=['Name','c','a','b'])
            dcc_dict[i]=dcc_df
        
            #Store marginal and copula degrees of freedom by period
            copula_dfs_dict[i]=copula_dfs_period
        
        #CONDITIONAL CORRELATION PLOTS
        #we plot the entire period DCC for each index, only once the dfs have been calibrated
        if calibration_flag==False:
            #prepare data in new dataframe that has dates as index
            period='all'
            plot_r_t=r_t_dict[period]   
            plot_list=[]
            for d_ in range(0,len(dates)): 
                #we create a dataframe with DCC of each index (they were obtained separately)
                partial=[dates[d_]]
                for n in range(1,n_):
                    if i_1!=0:
                        if n==1:
                            continue
                    partial.append(plot_r_t[stock_names[n]][d_])
                plot_list.append(partial)
            
            if i_1==1:
                plot_df = pd.DataFrame(plot_list, columns=['Date', 'BBBOND','MSCI','SPGSEN','SP5IAIR','SP5EHCR','SPEUIT']) 
            if i_1==0:
                plot_df = pd.DataFrame(plot_list, columns=['Date','SOLGB','BBBOND','MSCI','SPGSEN','SP5IAIR','SP5EHCR','SPEUIT']) 
            
            plot_df=plot_df.set_index('Date')
            
            for n in range(1,n_):
                if i_1!=0:
                    if n==1:
                        continue
                
                fig = plt.figure()
                plt.figure(figsize=((15,5)))
                #plt.title('Conditional correlation {} - {}'.format(stock_names[i_1],stock_names[n]))
                #plt.grid()
                #plt.figure(figsize=(15,5))
                #plt.title('Conditional correlation {} - {}'.format(stock_names[i_1],stock_names[n]),fontweight='bold')
                plt.grid()
                plt.plot(plot_df[stock_names[n]], linewidth=1.5)
                plt.yticks(fontsize=21)
                plt.xticks(fontsize=21)
                if i_1==0:
                     plt.savefig('plots/dcc/{}{}.png'.format(n,'a'),bbox_inches='tight') 
                else:
                    plt.savefig('plots/dcc/{}.png'.format(n-1), bbox_inches='tight')
         
        
        #OUTPUT EXCEL TABLES
        #Lower and upper tail dependence
        #they don't change with the dfs so we only compute them at the calibration stage
        if calibration_flag==True:
            if i_1==0:
                with pd.ExcelWriter('excel_tables/partialtables/Tail_dep_0.xlsx') as writer:
                    for k in tail_dep_dict:
                        pd.DataFrame(tail_dep_dict[k]).to_excel(writer, sheet_name='%s' % k)
            elif i_1==1:
                with pd.ExcelWriter('excel_tables/partialtables/Tail_dep_1.xlsx') as writer:
                    for k in tail_dep_dict:
                        pd.DataFrame(tail_dep_dict[k]).to_excel(writer, sheet_name='%s' % k)
            
        #Unconditional correlation
        #Changes with the dfs
        if i_1==0:
            if calibration_flag==True:
                with pd.ExcelWriter('excel_tables/partialtables/uncond_corr_df4_0.xlsx') as writer:
                    for k in uncond_corr_dict:
                        pd.DataFrame(uncond_corr_dict[k]).to_excel(writer, sheet_name='%s' % k)
            else:
                with pd.ExcelWriter('excel_tables/partialtables/uncond_corr_0.xlsx') as writer:
                    for k in uncond_corr_dict:
                        pd.DataFrame(uncond_corr_dict[k]).to_excel(writer, sheet_name='%s' % k)
                    for k in r_t_dict:
                        pd.DataFrame(r_t_dict[k]).to_excel(writer, sheet_name='dcc_%s' % k)
        elif i_1==1:
            if calibration_flag==True:
                with pd.ExcelWriter('excel_tables/partialtables/uncond_corr_df4_1.xlsx') as writer:
                    for k in uncond_corr_dict:
                        pd.DataFrame(uncond_corr_dict[k]).to_excel(writer, sheet_name='%s' % k)
            else:
                with pd.ExcelWriter('excel_tables/partialtables/uncond_corr_1.xlsx') as writer:
                    for k in uncond_corr_dict:
                        pd.DataFrame(uncond_corr_dict[k]).to_excel(writer, sheet_name='%s' % k)
                    for k in r_t_dict:
                        pd.DataFrame(r_t_dict[k]).to_excel(writer, sheet_name='dcc_%s' % k)
        
        #Export parameters of marginal distributions and GARCH parameters
        #and copula degrees of freedom
        #Once the degrees of freedom are correct
        if calibration_flag==False:
            if i_1==0:
                dcc_dict_BBGB=dcc_dict
                #save marginal parameters
                with pd.ExcelWriter('excel_tables/partialtables/marginal_0.xlsx') as writer:
                    mu_temp=pd.DataFrame([mu_marg_dict[k] for k in mu_marg_dict], index=[k for k in mu_marg_dict])
                    sigma2_temp=pd.DataFrame([sigma2_marg_dict[k] for k in sigma2_marg_dict], index=[k for k in sigma2_marg_dict])
                    marginal_df_temp=pd.DataFrame([marginal_dfs_dict[k] for k in marginal_dfs_dict], index=[k for k in marginal_dfs_dict])
                    marginal_df_temp.columns=stock_names
                    
                    mu_temp.to_excel(writer, sheet_name='location')
                    sigma2_temp.to_excel(writer, sheet_name='scale')
                    marginal_df_temp.to_excel(writer, sheet_name='dfs')
                    
                    for k in garch_dict:    
                        pd.DataFrame(garch_dict[k]).to_excel(writer, sheet_name='GARCH %s' %k) 
                        #although it doesn't change with i_1
                    for k in dcc_dict:    
                        pd.DataFrame(dcc_dict[k]).to_excel(writer, sheet_name='DCC %s' %k) 
                #save copula dfs
                with pd.ExcelWriter('excel_tables/partialtables/dfs_0.xlsx') as writer:
                    copula_df_temp=pd.DataFrame([copula_dfs_dict[k] for k in copula_dfs_dict], index=[k for k in copula_dfs_dict])
                    copula_df_temp.columns=stock_names
                    copula_df_temp.to_excel(writer, sheet_name='copula dfs')
            elif i_1==1:
                dcc_dict_SOLGB=dcc_dict
                #save marginal parameters
                with pd.ExcelWriter('excel_tables/partialtables/marginal_1.xlsx') as writer:
                    mu_temp=pd.DataFrame([mu_marg_dict[k] for k in mu_marg_dict], index=[k for k in mu_marg_dict])
                    sigma2_temp=pd.DataFrame([sigma2_marg_dict[k] for k in sigma2_marg_dict], index=[k for k in sigma2_marg_dict])
                    marginal_df_temp=pd.DataFrame([marginal_dfs_dict[k] for k in marginal_dfs_dict], index=[k for k in marginal_dfs_dict])
                    marginal_df_temp.columns=stock_names
                    
                    mu_temp.to_excel(writer, sheet_name='location')
                    sigma2_temp.to_excel(writer, sheet_name='scale')
                    marginal_df_temp.to_excel(writer, sheet_name='dfs')
                    for k in garch_dict:    
                        pd.DataFrame(garch_dict[k]).to_excel(writer, sheet_name='GARCH %s' %k)
                    for k in dcc_dict:    
                        pd.DataFrame(dcc_dict[k]).to_excel(writer, sheet_name='DCC %s' %k) 
                #save copula dfs
                with pd.ExcelWriter('excel_tables/partialtables/dfs_1.xlsx') as writer:
                    copula_df_temp=pd.DataFrame([copula_dfs_dict[k] for k in copula_dfs_dict], index=[k for k in copula_dfs_dict])
                    copula_df_temp.columns=stock_names
                    copula_df_temp.to_excel(writer, sheet_name='copula dfs')
        
    #Now that the setup with 4 degrees of freedom has been run, still within the calibration 
    #framework (calibration_flag=True) the calibration of the degrees of freedom 
    #is performed by optimizing the dots ratio
    if calibration_flag==True:
        dots_ratio_calibration(n_,k_,tau_,subperiods)


    ################CALIBRATION NEW: based on MLE
    




#Test adequacy of GARCH(1,1) with Li-Mak (1994) test.
#Low p-values reject H0 of no autocorrelation in squared standardized residuals
#and thus indicate that there are remaining ARCH effects in the data
#which the GARCH(1,1) model did not adequately capture.
times=[i for i in subperiods][1:]
pvals=np.zeros((len(times),n_))
c=0
for j in times:
    for i in range(n_):
        pvals[c,i], method = LiMakTest(eps_dict[j][:,i],sigma2_dict[j][:,i],fitdf=1,weighted=False,lag=2)
    c+=1
test=pvals<0.01
#print(test)
if test.any():
    print(method + ":\nGARCH(1,1) is not adequate (99% significance level).")
else:
    print(method + ":\nGARCH(1,1) is adequate.")




#########################################################################################
#Set 'boostrap' to True if want to bootstrap the confidence interval of the 
#garch and dcc parameters

bootstrap=True
if bootstrap==True:
    #Calling the functions for the bootstrap
    nsims=1000
    ci_dict_garch, pvals_dict_garch = {},{} 
    #different dictionaries for dcc parameters calibrated against BBGB and SOLGB
    ci_dict_dcc_BBGB, pvals_dict_dcc_BBGB = {},{} 
    ci_dict_dcc_SOLGB, pvals_dict_dcc_SOLGB = {},{}
    df_stocks=pd.read_excel(path + '\\input_data.xlsx', index_col=0) 
    for i in subperiods:
        t_first = subperiods[i][0]    
        t_last=subperiods[i][1]
        dx=dx_all[df_stocks.index.get_loc(t_first):df_stocks.index.get_loc(t_last)+1,:]
        garch_period=garch_dict[i]
        dfs_period=marginal_dfs_dict[i]
        ci_dict_garch[i], pvals_dict_garch[i], boot_epsilon, p= conf_interval_garch_flex_p(dx,nsims,tau_,garch_period,dfs_period)
        
        
        #Different confidence intervals and pvalues for dcc parameters calibrated against BBGB and SOLGB
        #SBAGLIATI, SISTEMALI:
        
        # dcc_period=dcc_dict_BBGB[i].iloc[:,1:].to_numpy()
        # ci_dict_dcc_BBGB[i], pvals_dict_dcc_BBGB[i]=conf_interval_dcc(boot_epsilon, dcc_period, p, nu_dict, 0, i, stock_names)
        # dcc_period=dcc_dict_SOLGB[i].iloc[:,1:].to_numpy()
        # ci_dict_dcc_SOLGB[i], pvals_dict_dcc_SOLGB[i]=conf_interval_dcc(boot_epsilon, dcc_period, p, nu_dict, 1, i, stock_names)


    # #Exporting confidence intervals and pvalues:
    # with pd.ExcelWriter("excel_tables/partialtables/garch_bootstrap.xlsx") as writer:
    #     for k in ci_dict_garch:
    #         #confidence intervals dataframes
    #         df_ci_alpha=pd.DataFrame(ci_dict_garch[k][0,:,:]) 
    #         df_ci_beta=pd.DataFrame(ci_dict_garch[k][1,:,:]) 
    #         df_ci_omega=pd.DataFrame(ci_dict_garch[k][2,:,:]) 
    #         df_ci_mu=pd.DataFrame(ci_dict_garch[k][3,:,:]) 
    #         df_ci_alpha.index=stock_names
    #         df_ci_beta.index=stock_names
    #         df_ci_omega.index=stock_names
    #         df_ci_mu.index=stock_names
    #         #pvalues dataframe
    #         df_pvals=pd.DataFrame(pvals_dict_garch[k])
    #         df_pvals.columns=stock_names
    #         df_pvals.index=['alpha','beta','omega','mu']
    #         #export to Excel
    #         df_ci_alpha.to_excel(writer, sheet_name='ci_alpha_%s' %k)
    #         df_ci_beta.to_excel(writer, sheet_name='ci_beta_%s' %k)
    #         df_ci_omega.to_excel(writer, sheet_name='ci_omega_%s' %k)
    #         df_ci_mu.to_excel(writer, sheet_name='ci_mu_%s' %k)
    #         df_pvals.to_excel(writer, sheet_name='pvals_%s' %k)


