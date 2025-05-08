import numpy as np
import pandas as pd
import math
from scipy import linalg
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore',RuntimeWarning)


################ computes log-likelihood of student t distribution ########
def t_loglikelihood(x,mu,q2_t,df):   #d is the dimension of multivariate t
    if np.shape(q2_t) == (): #univariate
        l = stats.t.logpdf(x, df, mu, q2_t)
    else: #multivariate case
        d=q2_t.shape[0]
        sigma2 = (df - 2.) / df * q2_t
        l=math.lgamma((df+d)/2.)-math.lgamma(df/2)- \
            (d/2.)*np.log(math.pi*df)-0.5*np.log(np.linalg.det(sigma2))- \
            (df+d)/2.*np.log(1.+(np.dot(np.dot((x-mu).T,np.linalg.inv(sigma2)),(x-mu)))/df)
    return l
######### finds flexible probabilities with exponential decay ########
def flex_p(t_, half_life,t_star=None):
    if t_star ==None:
        t_star=t_ #we take the final observation as the target time t*
    p = [np.exp(-(np.log(2.)/half_life)*abs(t_star - i)) for i in np.arange(0, t_)]
    p = p / np.sum(p)
    return p

############# computes mean and covariance of a sample based on a vector of probabilities
def mean_and_cov(y, p=None):   ####
    if p is None:
        p = np.ones(len(y)) / len(y)
    e_y = np.dot(p,y)
    cov_y = ((y-e_y).T*p) @ (y-e_y)
    return e_y, cov_y

######## transforms covariance matrix into correlation matrix #########
def covariance_to_correlation(cov):
    std = np.sqrt(np.diag(cov))     #find standard deviations
    inv_diag_std = np.diag(1/std) #diagonal matrix with inverse std as entries
    #find correlation matrix
    corr = np.dot(np.dot(inv_diag_std,cov),inv_diag_std)
    return corr

#######calibration of dcc_model for gaussian and student t case######
def dcc_parameters(xi, p=None, rho2=None):  ##### NEW VERSION also finds DFS
    t_ = len(xi)
    n_ = xi.shape[1]
    if p is None:
        p = np.full(shape=t_, fill_value=1/t_) #equal probabilities unless otherwise provided
    if rho2 is None:  #method of moments estimator if not provided
        _, rho2 = mean_and_cov(xi, p)
        rho2 = covariance_to_correlation(rho2)
    param_initial = [0.001, 0.98, 4]  #initial guess for parameters a, b, nu
    #find (negative) log-likelihood of GARCH (to then minimize)
    def neg_llh_gauss(parameters):
        mu = np.zeros(n_) #they are standardized
        a, b = parameters[:-1]
        q2_t = rho2.copy() #initialize value as = q bar
        r2_t = covariance_to_correlation(q2_t)
        n_llh = 0.0 #initialize negative loglikelihood
        for t in range(t_):
            n_llh = n_llh - p[t] * multivariate_normal.logpdf(xi[t, :], mu, r2_t)
            q2_t = rho2 * (1 - a - b) + a * np.outer(xi[t, :], xi[t, :]) + b * q2_t
            r2_t = covariance_to_correlation(q2_t)
        return n_llh
    def neg_llh_student(parameters):
        mu = np.zeros(n_) #they are standardized
        a, b, nu = parameters
        q2_t = rho2.copy() #initialize value as = q bar
        r2_t = covariance_to_correlation(q2_t)
        n_llh = 0.0 #initialize negative loglikelihood
        for t in range(t_):
            n_llh = n_llh - p[t] * t_loglikelihood(xi[t, :], mu=mu, q2_t=q2_t, df=nu)   #####
            q2_t = rho2 * (1 - a - b) + a * np.outer(xi[t, :], xi[t, :]) + b * q2_t
            r2_t = covariance_to_correlation(q2_t)
        return n_llh
    #minimize neg. log-likelihood
    #set boundaries
    bounds = ((1e-21, 1.), (1e-21, 1.), (4,None))
    #we impose a stationary constraint
    cons = {'type': 'ineq', 'fun': lambda param: 0.99 - param[0] - param[1]}
    #we find minimizer parameters based on whether data is gaussian or follows student t
    a, b, nu = minimize(neg_llh_student, param_initial, bounds=bounds, constraints=cons)['x']
    # if nu<1000:
    #     a, b = minimize(neg_llh_student, param_initial, bounds=bounds, constraints=cons)['x']
    # else:
    #     a, b = minimize(neg_llh_gauss, param_initial, bounds=bounds, constraints=cons)['x']
    #compute realized correlations and residuals
    q2_t = rho2.copy() #initialize value as = q bar
    r2_t = np.zeros((t_, n_, n_)) #there are t_   n_xn_  matrices of dcc
    r2_t[0, :, :] = covariance_to_correlation(q2_t) #the first one is the initial value
    for t in range(1, t_):
        q2_t = rho2 * (1 - a - b) + a * np.outer(xi[t, :], xi[t, :]) + b * q2_t
        r2_t[t, :, :] = covariance_to_correlation(q2_t)
    l_t = np.linalg.cholesky(r2_t)
    epsi = np.linalg.solve(l_t, xi)
    return [1. - a - b, a, b], r2_t, epsi, q2_t, nu

def dcc_parameters_old(xi, p=None, rho2=None,nu=4):  #####OLD VERSION
    t_ = len(xi)
    n_ = xi.shape[1]
    if p is None:
        p = np.full(shape=t_, fill_value=1/t_) #equal probabilities unless otherwise provided
    if rho2 is None:  #method of moments estimator if not provided
        _, rho2 = mean_and_cov(xi, p)
        rho2 = covariance_to_correlation(rho2)
    param_initial = [0.01, 0.99]  #initial guess for parameters
    #find (negative) log-likelihood of GARCH (to then minimize)
    def neg_llh_gauss(parameters):
        mu = np.zeros(n_) #they are standardized
        a, b = parameters
        q2_t = rho2.copy() #initialize value as = q bar
        r2_t = covariance_to_correlation(q2_t)
        n_llh = 0.0 #initialize negative loglikelihood
        for t in range(t_):
            n_llh = n_llh - p[t] * multivariate_normal.logpdf(xi[t, :], mu, r2_t)
            q2_t = rho2 * (1 - a - b) + a * np.outer(xi[t, :], xi[t, :]) + b * q2_t
            r2_t = covariance_to_correlation(q2_t)
        return n_llh
    def neg_llh_student(parameters):
        mu = np.zeros(n_) #they are standardized
        a, b = parameters
        q2_t = rho2.copy() #initialize value as = q bar
        r2_t = covariance_to_correlation(q2_t)
        n_llh = 0.0 #initialize negative loglikelihood
        for t in range(t_):
            n_llh = n_llh - p[t] * t_loglikelihood(xi[t, :], mu=mu, q2_t=q2_t, df=nu)   #####
            q2_t = rho2 * (1 - a - b) + a * np.outer(xi[t, :], xi[t, :]) + b * q2_t
            r2_t = covariance_to_correlation(q2_t)
        return n_llh
    #minimize neg. log-likelihood
    #set boundaries
    bounds = ((1e-21, 1.), (1e-21, 1.))
    #we impose a stationary constraint
    cons = {'type': 'ineq', 'fun': lambda param: 0.99 - param[0] - param[1]}
    #we find minimizer parameters based on whether data is gaussian or follows student t
    if nu<1000:
        a, b = minimize(neg_llh_student, param_initial, bounds=bounds, constraints=cons)['x']
    else:
        a, b = minimize(neg_llh_gauss, param_initial, bounds=bounds, constraints=cons)['x']
    #compute realized correlations and residuals
    q2_t = rho2.copy() #initialize value as = q bar
    r2_t = np.zeros((t_, n_, n_)) #there are t_   n_xn_  matrices of dcc
    r2_t[0, :, :] = covariance_to_correlation(q2_t) #the first one is the initial value
    for t in range(1, t_):
        q2_t = rho2 * (1 - a - b) + a * np.outer(xi[t, :], xi[t, :]) + b * q2_t
        r2_t[t, :, :] = covariance_to_correlation(q2_t)
    l_t = np.linalg.cholesky(r2_t)
    epsi = np.linalg.solve(l_t, xi)
    return [1. - a - b, a, b], r2_t, epsi, q2_t



#################principal component analysis and spectral decomposition###############
def spectral_dec_cov(cov, k=None):     #principal component analysis
    i=len(cov)
    if k is None: #number of factors for factor analysis
        k = len(cov)
    evals, evecs = linalg.eigh(cov, eigvals=(i-k, i-1)) #are returned in ascending order
    evals = np.flip(evals)  #reverse order
    evecs = np.fliplr(evecs) #reverse order
    #we impose a sign convention ensuring that the largest element in each eigenvector is positive
    largest = np.argmax(abs(evecs), axis=0) #find position of largest in magnitude
    largest = np.diag(evecs[largest, :]) < 0 #find if they are <0
    evecs[:, largest] = -evecs[:, largest] #change sign of the ones <0
    return evecs, evals

#########mahalanobis distance of a vector from a vector of average values########
def mah_dist(x, mu, s2):     ######
    if np.ndim(x) > 1:
        m_d = np.sqrt(np.sum((x-mu).T * np.dot(np.linalg.inv(s2),(x-mu).T), axis=0))
    else:
        m_d = np.sqrt((x-mu)/s2*(x-mu))
    return m_d

#############finds parameters of marginals #####
def fit_marginals_flex_p(epsilon, p=None, nu=1000, threshold=1e-3):                ####
    if len(epsilon.shape) == 1:
        epsilon = epsilon.reshape(-1, 1)
    t_ = len(epsilon)
    n_ = epsilon.shape[1]
    if p is None:
        p = np.full(shape=t_, fill_value=1/t_) #equal probabilities unless otherwise provided
    # initial values with method of moments
    mu, sigma2 = mean_and_cov(epsilon, p)
    if nu > 2.: #otherwise covariance is not defined
        sigma2 = (nu - 2.)/nu*sigma2
    for i in range(1000):  #1000 iterations
        if nu >= 1e3 and np.linalg.det(sigma2) < 1e-13:
            w = np.ones(t_)
        else:
            w = (nu + n_) / (nu + mah_dist(epsilon, mu, sigma2) ** 2)
        q = w*p
        #update location and dispersion parameters
        mu_old, sigma2_old = mu, sigma2
        mu, sigma2 = mean_and_cov(epsilon, q)
        mu = mu/np.sum(q)
        #repeat until convergence in infinity norm
        #(error goes below the threshold for both estimates)
        err = max(np.linalg.norm(mu - mu_old, ord=np.inf) /
                  np.linalg.norm(mu_old, ord=np.inf),
                  np.linalg.norm(sigma2 - sigma2_old, ord=np.inf) /
                  np.linalg.norm(sigma2_old, ord=np.inf))
        if err <= threshold:
            break
    return np.squeeze(mu), np.squeeze(sigma2)

########estimation of garch parameters with flexible probabilities######
def garch_flex_p(dx, p=None, sigma2_0=None, rescale=False):                ########
    t_ = len(dx)
    #set probabilities
    if p is None:
        p = np.full(shape=t_, fill_value=1/t_) #equal probabilities unless otherwise provided
    #compute sample mean and variance
    if rescale is True:     
        mu, s2 = mean_and_cov(dx, p)
    #if not provided, set initial variance
    if sigma2_0 is None:
        p_tau = flex_p(t_, t_/3, 0)
        _, sigma2_0 = mean_and_cov(dx, p_tau)
    #we define initial parameters and impose a stationarity constraint
    param_0 = [0.01, 0.94, 0.05*s2, mu]
    #and rescale data, if requested
    if rescale is True:
        dx = (dx - mu) / np.sqrt(s2)
        param_0[2] = param_0[2] / s2
        param_0[3] = param_0[3] - mu
        sigma2_0 = sigma2_0 / s2
    #we compute negative log-likelihood of GARCH
    def neg_loglikelihood(parameters):
        alpha, beta, omega, mu = parameters
        sigma2 = sigma2_0 #initial value
        neg_l = 0.0
        for t in range(t_):
            if np.abs(sigma2) > 1e-100:     #avoid too small values of sigma for convergence issues
                neg_l = neg_l + p[t]*(np.log(sigma2) + (dx[t]-mu)**2 /sigma2) #normal loglikelihood 
                        #(literature shows it can be used to approximate student t case with 
                        #very little change to parameter values)
                        #J.H. VENTER and P.J.. JONGH (2002). Risk estimation using the normal inverse
                        #gaussian distribution. Journal of Risk, Vol. 4, pages 1–24
                        #M. BERG JENSEN and A. LUNDE (2001). The nig-s and arch model: A fat-tailed
                        #stochastic, and autoregressive conditional heteroscedastic volatility model. Working
                        #paper series No. 83, University of Aarhus.
                
            sigma2 = omega + alpha * (dx[t]- mu) ** 2 + beta * sigma2
        return neg_l
    #set boundaries fpr parameters
    bnds = ((1e-21, 1.), (1e-21, 1.), (1e-21, None), (None, None))
    #impose stationary constraint on alpha and beta
    cons = {'type': 'ineq', 'fun': lambda param: 0.95 - param[0] - param[1]}
    alpha_hat, beta_hat, omega_hat, mu_hat = minimize(neg_loglikelihood, param_0, bounds=bnds, constraints=cons)['x']
    #compute realized values
    sigma2_hat = np.full(t_, sigma2_0) #will only keep first     one
    for t in range(1,t_):
        sigma2_hat[t] = omega_hat + alpha_hat*(dx[t-1]-mu_hat)**2 + beta_hat*sigma2_hat[t-1]
    epsilon = (dx - mu_hat) / (sigma2_hat**0.5)
    #invert rescaling, if it had been imposed
    if rescale is True:
        omega_hat = omega_hat * s2
        mu_hat = mu+ mu_hat * s2**0.5
        sigma2_hat = sigma2_hat * s2
    return np.array([alpha_hat, beta_hat, omega_hat, mu_hat]), sigma2_hat, np.squeeze(epsilon)
    
##########GARCH with student t increments
def garch_t_flex_p(dx, p=None, sigma2_0=None, dfs=None, rescale=False):                ########
    t_ = len(dx)
    #set probabilities
    if p is None:
        p = np.full(shape=t_, fill_value=1/t_) #equal probabilities unless otherwise provided
    #compute sample mean and variance
    if rescale is True:     
        mu, s2 = mean_and_cov(dx, p)
    #if not provided, set initial variance
    if sigma2_0 is None:
        p_tau = flex_p(t_, t_/3, 0)
        _, sigma2_0 = mean_and_cov(dx, p_tau)
    #we define initial parameters and impose a stationarity constraint
    if dfs is None:
        param_0 = [0.01, 0.94, 0.05*s2, mu, 3]
    else:
        param_0 = [0.01, 0.94, 0.05*s2, mu]
    #and rescale data, if requested
    if rescale is True:
        dx = (dx - mu) / np.sqrt(s2)
        param_0[2] = param_0[2] / s2
        param_0[3] = param_0[3] - mu
        sigma2_0 = sigma2_0 / s2
    #we compute negative log-likelihood of GARCH
    if dfs is None:
        def neg_loglikelihood(parameters):
            alpha, beta, omega, mu, df = parameters
            sigma2 = sigma2_0 #initial value
            #neg_l = 0.0
            neg_l_st = 0.0
            for t in range(t_):
                if np.abs(sigma2) > 1e-100:     #avoid too small values of sigma for convergence issues
                    #neg_l = neg_l + p[t]*0.5*(np.log(2* math.pi)+np.log(sigma2) + (dx[t]-mu)**2 /sigma2) #normal loglikelihood 
                    neg_l_st= neg_l_st - p[t]*stats.t.logpdf(dx[t], df, mu, ((df-2)*sigma2/df)**0.5)
                sigma2 = omega + alpha * (dx[t]- mu) ** 2 + beta * sigma2
            return neg_l_st
    else:
        df=dfs
        def neg_loglikelihood(parameters):
            alpha, beta, omega, mu = parameters
            sigma2 = sigma2_0 #initial value
            #neg_l = 0.0
            neg_l_st = 0.0
            for t in range(t_):
                if np.abs(sigma2) > 1e-100:     #avoid too small values of sigma for convergence issues
                    #neg_l = neg_l + p[t]*0.5*(np.log(2* math.pi)+np.log(sigma2) + (dx[t]-mu)**2 /sigma2) #normal loglikelihood 
                    neg_l_st= neg_l_st - p[t]*stats.t.logpdf(dx[t], df, mu, ((df-2)*sigma2/df)**0.5)
                sigma2 = omega + alpha * (dx[t]- mu) ** 2 + beta * sigma2
            return neg_l_st
    #set boundaries fpr parameters
    if dfs is None:
        bnds = ((1e-21, 1.), (1e-21, 1.), (1e-21, None), (None, None), (3,None))
    else:
        bnds = ((1e-21, 1.), (1e-21, 1.), (1e-21, None), (None, None))
    #impose stationary constraint on alpha and beta
    cons = {'type': 'ineq', 'fun': lambda param: 0.95 - param[0] - param[1]}
    if dfs is None:
        alpha_hat, beta_hat, omega_hat, mu_hat, df_hat = minimize(neg_loglikelihood, param_0, bounds=bnds, constraints=cons)['x']
    else:
        alpha_hat, beta_hat, omega_hat, mu_hat = minimize(neg_loglikelihood, param_0, bounds=bnds, constraints=cons)['x']
        df_hat=np.array(0)
    #compute realized values
    sigma2_hat = np.full(t_, sigma2_0) #will only keep first     one
    for t in range(1,t_):
        sigma2_hat[t] = omega_hat + alpha_hat*(dx[t-1]-mu_hat)**2 + beta_hat*sigma2_hat[t-1]
    epsilon = (dx - mu_hat) / (sigma2_hat**0.5)
    #invert rescaling, if it had been imposed
    if rescale is True:
        omega_hat = omega_hat * s2
        mu_hat = mu+ mu_hat * s2**0.5
        sigma2_hat = sigma2_hat * s2
    return np.array([alpha_hat, beta_hat, omega_hat, mu_hat]), sigma2_hat, np.squeeze(epsilon), df_hat
    

########DEGREES OF FREEDOM CALIBRATION############
# def df_calibration():
#     df_0=3 #initial parameter value
    
#     def neg_loglikelihood(df):
#         df = df_0 #initial value
#         neg_l = 0.0
#         for t in range(t_):
#             if np.abs(sigma2) > 1e-100: #avoid too small values of sigma for convergence issues
#                 #neg_l = neg_l + p[t]*0.5*(np.log(2* math.pi)+np.log(sigma2) + (dx[t]-mu)**2 /sigma2) #normal loglikelihood 
#                 neg_l= neg_l - p[t]*stats.t.logpdf(dx[t], df, mu, ((df-2)*sigma2/df)**0.5)
#             sigma2 = omega + alpha * (dx[t]- mu) ** 2 + beta * sigma2
#         return neg_l
    
    
#     #set boundaries for parameter
#     bnds = (3,None)
#     #minimize
#     df_hat = minimize(neg_loglikelihood, df_0, bounds=bnds)['x']
#     return df_hat




##############################factor analysis#######################
def factor_analysis(cov, k_):
    #with principal axis factorization algorithm
    n_ = len(cov)
    S = np.eye(n_) #matrix of zeros with ones on diagonal
    loadings = np.full((n_,k_),0.7) #initial guess  
    delta2_old = np.zeros(n_) #to store specific-factor variances
    loadings_old = np.full((n_,k_),0.7) #initial guess  
    for i in range(100):
        delta2 = np.diag(cov - loadings @ loadings.T) #updated specific-factor variances
        evecs, evals = spectral_dec_cov(np.linalg.lstsq(S, (np.linalg.lstsq(S, cov - np.diag(delta2), rcond=None)[0]).T , rcond=None)[0], k_)
        loadings = S @ (evecs @ np.diag(evals**0.5)) #update loadings
        if np.max(np.abs(loadings - loadings_old))/max(np.max(np.abs(loadings_old)), 1e-20) and np.max(np.abs(delta2 - delta2_old))/max(np.max(np.abs(delta2_old)), 1e-20) < 1e-20:
            break  #accounts for convergence
        loadings_old = loadings
        delta2_old = delta2         
    return loadings, delta2


#############plot expectation-covariance ellipsoids#########
def exp_cov_ellipsoid(mu, s2, color='k'):
    evecs, evals = spectral_dec_cov(s2)
    diag_evals = np.diagflat(np.sqrt(np.maximum(evals, 0))) 
    #find points of circle with unit radius
    y = [np.cos(np.arange(0, 2*math.pi, (2*math.pi)/1500)),\
         np.sin(np.arange(0, 2*math.pi, (2*math.pi)/1500))]
    #find the points of the ellipse as an affine transformation
    x = diag_evals@y      #square root of evals are axis half-lengths
    x = evecs@x #eigenvectors are ellipse axis direction
    x = mu.reshape((2, 1)) + x #shift ellipse by mean vector
    plt.plot(x[0], x[1], lw=2, color=color) #plot ellipse
    plt.grid(True)
    return x.T

#####################################################################
import random

def garch_flex_p_bootstrap(dx,nreps,tau_,dfs_period=None):
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
    boot_param=np.zeros((nreps,4,n))
    boot_sigma2=np.zeros((nreps,t,n))
    boot_epsilon=np.zeros((nreps,t,n))
    boot_dfs=np.zeros((nreps,n))
    for rep in range(nreps): #loop over number of simulations
        #Prepare arrays to store values
        param = np.zeros((4, n)) #parameters a,b,omega,mu
        sigma2 = np.zeros((t, n)) #conditional variances
        dfs = np.zeros(n) #marginal dfs
        epsilon = np.zeros((t, n)) #innovations
        for i in range(n): #loop over number of assets
            param[:, i], sigma2[:, i], epsilon[:, i], dfs[i] = garch_t_flex_p(sim_logs[rep,:, i], p, dfs= dfs_period, rescale=True)
        boot_param[rep,:,:]=param #store the values of all simulations
        boot_sigma2[rep,:,:]=sigma2
        boot_epsilon[rep,:,:]=epsilon
        boot_dfs[rep,:]=dfs
        print(rep)
    return sim_logs, boot_param, boot_sigma2, boot_epsilon, p, boot_dfs

def dcc_function(epsilon,p,nu_dict,i_1,period,stock_names,k_=2):
    n=np.shape(epsilon)[1]   
    t_=np.shape(epsilon)[0]
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
        u = stats.t.cdf(epsilon[:, j], df=nu_dict[period][i_1][j], loc=mu_marg[j],
                  scale=np.sqrt(sigma2_marg[j]))
        u[u <= 10**(-7)] = 10**(-7)
        u[u >= 1-10**(-7)] = 1-10**(-7)
        xi[:, j] = stats.t.ppf(u, df=nu_dict[period][i_1][j]) 
    #LOOP OVER STOCKS (same i_1 and a different i_2)
    for i_2 in range(i_1+1,n):
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
    return dcc_df
    
    
def conf_interval_dcc(boot_epsilon,dcc_period,p,nu_dict,i_1,period,stock_names):    #outside loop over i_1 and period
    nreps=np.shape(boot_epsilon)[0]
    n=np.shape(boot_epsilon)[2]
    boot_dcc=np.zeros((nreps,n-i_1-1,3))
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
    c_is=np.zeros((3,n-i_1-1,2))
    pvals=np.zeros((n-i_1-1,3))
    #sort simulated values for each parameter
    boot_dcc_sorted=np.sort(boot_dcc,axis=0) #every row is a simulation. columns are 8 indices
    #compute confidence intervals of c, a, b
    c_is[0,:,:]=[(boot_dcc_sorted[quantile_low,j,0],boot_dcc_sorted[quantile_up,j,0]) for j in range(n-i_1-1)]
    c_is[1,:,:]=[(boot_dcc_sorted[quantile_low,j,1],boot_dcc_sorted[quantile_up,j,1]) for j in range(n-i_1-1)]
    c_is[2,:,:]=[(boot_dcc_sorted[quantile_low,j,2],boot_dcc_sorted[quantile_up,j,2]) for j in range(n-i_1-1)]
    #compute pvalues of c, a, b
    pvals[:,0]=np.count_nonzero(boot_dcc[:,:,0]>dcc_0[:,0],axis=0)/nreps
    pvals[:,1]=np.count_nonzero(boot_dcc[:,:,1]>dcc_0[:,1],axis=0)/nreps
    pvals[:,2]=np.count_nonzero(boot_dcc[:,:,2]>dcc_0[:,2],axis=0)/nreps
    pvals_df=pd.DataFrame(pvals,index=indices,columns=['c','a','b'])
    return c_is, pvals_df


def conf_interval_garch_flex_p(dx,nsims,tau_,garch_period, dfs_period=None):
    quantile_low=int(np.floor(0.024*nsims)) #for confidence interval
    quantile_up=int(np.floor(0.974*nsims)) #for confidence interval
    n=np.shape(dx)[1]
    if dfs_period is None: #then we also need to bootstrap dfs
        c_is=np.zeros((5,n,2))
        pvals=np.zeros((5,n))
    else: #then we don't need to bootstrap dfs
        c_is=np.zeros((4,n,2))
        pvals=np.zeros((4,n))
    sim_logs, boot_param, _, boot_epsilon, p, boot_dfs = garch_flex_p_bootstrap(dx, nsims, tau_)
    #get estimated parameter values from original sample
    alpha_0=garch_period[0,:]
    beta_0=garch_period[1,:]
    omega_0=garch_period[2,:]
    mu_0=garch_period[3,:]
    dfs_0=dfs_period
    #sort simulated values for each parameter
    alpha_sim=np.sort(boot_param[:,0,:],axis=0) #every row is a simulation. columns are 8 indices
    beta_sim=np.sort(boot_param[:,1,:],axis=0)
    omega_sim=np.sort(boot_param[:,2,:],axis=0)
    mu_sim=np.sort(boot_param[:,3,:],axis=0)
    dfs_sim=np.sort(boot_dfs,axis=0)
    #compute confidence intervals
    c_is[0,:,:]=[(alpha_sim[quantile_low,j],alpha_sim[quantile_up,j]) for j in range(n)]
    c_is[1,:,:]=[(beta_sim[quantile_low,j],beta_sim[quantile_up,j]) for j in range(n)]
    c_is[2,:,:]=[(omega_sim[quantile_low,j],omega_sim[quantile_up,j]) for j in range(n)]
    c_is[3,:,:]=[(mu_sim[quantile_low,j],mu_sim[quantile_up,j]) for j in range(n)]
    if dfs_period is None: 
        c_is[4,:,:]=[(dfs_sim[quantile_low,j],dfs_sim[quantile_up,j]) for j in range(n)]
    #compute pvalues
    pvals[0,:]=np.count_nonzero(alpha_sim>alpha_0,axis=0)/nsims
    pvals[1,:]=np.count_nonzero(beta_sim>beta_0,axis=0)/nsims 
    pvals[2,:]=np.count_nonzero(omega_sim>omega_0, axis=0)/nsims
    pvals[3,:]=np.count_nonzero(mu_sim>mu_0,axis=0)/nsims
    if dfs_period is None: 
        pvals[4,:]=np.count_nonzero(dfs_sim>dfs_0,axis=0)/nsims
    return c_is , pvals, boot_epsilon, p



    

