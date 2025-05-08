import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import gamma 

def LiMakTest(x,sigma2,lag=1,fitdf = 1, weighted=True):
    """
    Arguments:
        x: residuals
        sigma2: conditional variances
        lag: the statistic will be based on lag autocorrelation coefficients
        fitdf: the number of ARCH parameters fit to the model, default=1
        weighted: TRUE if a weighting scheme is to be used. FALSE for Li and Mak (1994) test
    """
    if type(x) != 'pandas.core.frame.DataFrame':
        x=pd.DataFrame(x)
    if type(sigma2) != 'pandas.core.frame.DataFrame':
        sigma2=pd.DataFrame(sigma2)
    x=x.dropna()
    sigma2=sigma2.dropna()
    #Checking for errors
    
    try:
        if x.shape[1] > 1:
          print("x is not a vector or univariate time series")
    except:
        j=0
    if fitdf >= lag:
      print("Lag must exceed fitted degrees of freedom")
    if fitdf < 1:
      print("Fitted degrees of freedom must be positive")
    if len(x)!=len(sigma2):
      print("Length of x and sigma2 must match")
    
    x=x**2/sigma2
    
    if weighted==True:
        method="Weighted Li-Mak test on autocorrelations (Gamma Approximation)"
    else:
        method="Li-Mak test on autocorrelations (Chi-Squared Approximation)"
      
    n=len(x)
    
    cor = sm.tsa.stattools.acf(x, nlags = lag,fft=False)  #check if same as r
    obs = cor[1:(lag + 2)]
    
    if weighted:
        weights = [(lag - i + (fitdf+1))/lag for i in range(fitdf+1,lag+1)]
        obs = obs[(fitdf):lag] #
        statistic= n * sum(weights*obs**2)
        #Weighted X-squared on Squared Residuals for fitted ARCH process
        shape = (3./4)*(lag + fitdf + 1)**2*(lag - fitdf)/(2*lag**2 + 3*lag + 2*lag*fitdf + 2*fitdf**2 + 3*fitdf + 1)
        scale = (2./3)*(2*lag**2 + 3*lag + 2*lag*fitdf + 2*fitdf**2 + 3*fitdf + 1)/(lag*(lag + fitdf + 1));
        parameter = pd.DataFrame((shape, scale))
        parameter.index= (("Shape", "Scale"))
        
    else:
        weights = [1 for i in range(0,lag-fitdf)] 
        obs = obs[(fitdf):lag] #
        statistic= n * sum(weights*obs**2)
        #X-squared on Squared Residuals for fitted ARCH process
        shape = (lag-fitdf)/2      # Chi-squared df in Gamma form.
        scale = 2
        parameter = lag-fitdf
        name_parameter = "Degrees of Freedom"

    #Approximate p-value
    pval = 1 - gamma.cdf(statistic, a=shape, scale=scale)


    return pval, method