import numpy as np
import pandas as pd
import math
import scipy.optimize as sco
import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore',DeprecationWarning)
warnings.simplefilter('ignore',PendingDeprecationWarning)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------- ASSET ALLOCATION STRATEGIES----------------------------------------------------------
# all 5 functions take as input the dataframe of asset returns and output the portfolio composition
# as an array of size equal to the number of assets

# 1 - MEAN VARIANCE STRATEGY
# ----------------------------------------------------------------------------------------------------------------------

def eff_portfolio(mean_returns,cov_matrix,target_return):
    cols=cov_matrix.columns
    bnds=tuple((0,1) for x in range(len(mean_returns)))
    con1= ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    con2= ({'type': 'eq', 'fun': lambda x:  target_return - np.sum(mean_returns*x)*252})
    x= sco.minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))*np.sqrt(252),
                          x0=np.tile(1/len(cols),len(cols)),bounds=bnds,constraints=(con1,con2),options={'disp': False})
    return x['x']

def eff_points(rendimenti_df, max_return=0.5):
    mean_returns = rendimenti_df.mean()
    cov_matrix = rendimenti_df.cov()  
    target=np.linspace(0,max_return,70)
    eff_weights=[]
    for i in target:
        eff_weights.append(eff_portfolio(mean_returns,cov_matrix,i))
    eff_points=np.zeros((len(eff_weights),2))
    for i in range(len(eff_weights)):
        eff_points[i][0]=np.sqrt(np.dot(eff_weights[i].T, np.dot(cov_matrix, eff_weights[i])))*np.sqrt(252)
        eff_points[i][1]=np.sum(mean_returns*eff_weights[i])*252
    return eff_points, eff_weights

def MeanVarianceStrategy(rendimenti_df,rf=0,max_return=0.5): #risk-free rate is rf
    eff_p, eff_weights = eff_points(rendimenti_df,max_return)
    sharpe_p=np.zeros(len(eff_p))  #compute sharpe ratio of all points on the efficient frontier
    for i in range(len(eff_p)):
        sharpe_p[i]=(eff_p[i,1]-rf)/eff_p[i,0]
    return eff_weights[np.argmax(sharpe_p)]

# 2 - MINIMUM VARIANCE STRATEGY
# ----------------------------------------------------------------------------------------------------------------------
def MinimumVarianceStrategy(dataframe_rendimenti):
    bound = 1.0
    rendimenti = np.array(dataframe_rendimenti)
    nn = np.shape(rendimenti)[0]
    m = np.shape(rendimenti)[1]

    sigma = np.zeros([m, m])
    for i in range(0, m, 1):
        for j in range(0, m, 1):
            sigma[i][j] = 252 * np.cov(rendimenti[:, i], rendimenti[:, j])[0][1]

    def risk(w):
        _w = w[:, np.newaxis]
        return np.dot(_w.transpose(), np.dot(sigma, _w)).reshape(1,)

    cons = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
    bnds = ((0, bound),) * sigma.shape[0]
    w_ini = np.repeat(1, np.shape(sigma)[0])
    w_ini = w_ini / sum(w_ini)
    res = sco.minimize(risk, w_ini, bounds=bnds, constraints=cons, options={'disp': False, 'ftol': 10 ** -10})
    return res['x']

# 3 - 1/N RULE
# ----------------------------------------------------------------------------------------------------------------------
def EqualWeight(dataframe_rendimenti):
    n = dataframe_rendimenti.shape[1]
    return np.full((n,),1/n)

# 4 - RISK PARITY
# ----------------------------------------------------------------------------------------------------------------------
def RiskParity(dataframe_rendimenti):
    bound = 1.0
    rendimenti = np.array(dataframe_rendimenti)
    nn = np.shape(rendimenti)[0]
    m = np.shape(rendimenti)[1]


    # matrice di covarianza
    sigma = np.zeros([m, m])
    for i in range(0, m, 1):
        for j in range(0, m, 1):
            sigma[i][j] = 252 * np.cov(rendimenti[:, i], rendimenti[:, j])[0][1]

    def riskparity_(x):
        n = len(sigma)
        w = np.mat(x).T
        port_var = np.sqrt(w.T * np.mat(sigma) * w)
        port_vec = np.mat(np.repeat(port_var / n, n)).T
        diag = np.mat(np.diag(x) / port_var)
        partial = np.mat(sigma) * w
        return np.square(port_vec - diag * partial).sum()

    cons = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
    bnds = ((0, bound),) * sigma.shape[0]
    w_ini = np.repeat(1, np.shape(sigma)[0])
    w_ini = w_ini / sum(w_ini)
    res = sco.minimize(riskparity_, w_ini, bounds=bnds, constraints=cons, options={'disp': False, 'ftol': 10 ** -10})
    return res['x']

# 5 - MINIMUM CVAR
# ----------------------------------------------------------------------------------------------------------------------
def Cvar_opt(dataframe_rendimenti, beta = 0.95):
    bound = 1.0
    ret = np.array(dataframe_rendimenti)

    q = np.shape(ret)[0]
    n = np.shape(ret)[1]
    m = 1 / (q * (1 - beta))

    c = np.array([1] + [m for k in range(q)] + [0 for k in range(n)])

    A1 = np.mat(np.eye(1 + q + n))
    A2 = np.hstack([np.mat(np.repeat(1., q)).T, np.eye(q), np.mat(np.array(ret))])
    A_ub = -np.vstack([A1, A2])
    b_ub = np.mat(np.repeat(0, 1 + 2 * q + n)).T

    A_eq = np.mat([0 for k in range(1 + q)] + [1 for k in range(n)])
    b_eq = np.mat(1.)

    bnds = ((0, np.inf),) * (1 + q) + ((0, bound),) * n
    linpro = sco.linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds, options={'disp': False})

    return linpro['x'][(1 + q):]

# 5 - MAXIMUM DIVERSIFICATION
# ----------------------------------------------------------------------------------------------------------------------
def MaxDiv(dataframe_rendimenti):
    rendimenti = np.array(dataframe_rendimenti)
    nn = np.shape(rendimenti)[0]
    m = np.shape(rendimenti)[1]
    sigma = np.zeros([m, m])
    for i in range(0, m, 1):
        for j in range(0, m, 1):
            sigma[i][j] = 252 * np.cov(rendimenti[:, i], rendimenti[:, j])[0][1]

    bound=1.

    diversification=lambda x: -np.dot(np.sqrt(np.diag(sigma)),x).sum()/np.sqrt(np.dot(x.T,np.dot(sigma,x)))
    cons=({'type':'eq', 'fun':lambda x: np.nansum(x)-1})
    w_ini=np.repeat(1,np.shape(sigma)[0])
    w_ini=w_ini/np.nansum(w_ini)
    bnds=((0,bound),)*np.shape(sigma)[0]
    result=sco.minimize(diversification,w_ini,bounds=bnds,constraints=cons,options={'disp': False,'ftol':10**-8})
    return result['x']



# ----------------------------------------------------------------------------------------------------------------------
# PORTFOLIO RETURNS
def rendimentiportafoglio(dataframe_rendimenti, pesi):
    rendimenti = np.array(dataframe_rendimenti)
    return np.dot(rendimenti, pesi)

# CUMULATED RETURNS
def rendimenticumulati(rendimenti):
    return (rendimenti + 1.0).cumprod()


# ----------------------------------------------------------------------------------------------------------------------
# TABELLE COMPOSIZIONE PORTAFOGLI
def pesiportafogli(dataframe_rendimenti):
    pesi_meanvar = MeanVarianceStrategy(dataframe_rendimenti)
    pesi_minvar = MinimumVarianceStrategy(dataframe_rendimenti)
    pesi_ew = EqualWeight(dataframe_rendimenti)
    pesi_rp = RiskParity(dataframe_rendimenti)
    pesi_cvar = Cvar_opt(dataframe_rendimenti)
    pesi_maxdiv = MaxDiv(dataframe_rendimenti)
    return pesi_meanvar, pesi_minvar, pesi_ew, pesi_rp, pesi_cvar, pesi_maxdiv


# ----------------------------------------------------------------------------------------------------
# IN-SAMPLE PROCEDURE
def insample_approach (rendimenti_df, displ=100,k=35):
    i = 0
    steps = math.floor((len(rendimenti_df[:-1]) - displ) / k)  
    n_asset = rendimenti_df.shape[1]
    l = k * steps
    dates=rendimenti_df.index[displ:displ+k*steps]
    w_meanvar_in, w_minvar_in, w_ew_in, w_rp_in, w_cvar_in, w_maxdiv_in = np.empty([steps, n_asset]), np.empty([steps, n_asset]), \
                                                                          np.empty([steps, n_asset]), np.empty([steps, n_asset]), \
                                                                          np.empty([steps, n_asset]), np.empty([steps, n_asset])
    RP_meanvar_in, RP_minvar_in, RP_ew_in, RP_rp_in, RP_cvar_in, RP_maxdiv_in = np.empty([l, ]), np.empty([l, ]), np.empty([l, ]), \
                                                                                np.empty([l, ]), np.empty([l, ]), np.empty([l, ])
    end = displ + k
    for i in range(steps):
        w_meanvar_in[i], w_minvar_in[i], w_ew_in[i], w_rp_in[i], w_cvar_in[i], w_maxdiv_in[i] = pesiportafogli(rendimenti_df[displ:end])
        RP_minvar_in[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[displ:end], w_minvar_in[i])
        RP_meanvar_in[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[displ:end], w_meanvar_in[i])

        RP_ew_in[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[displ:end], w_ew_in[i])
        RP_rp_in[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[displ:end], w_rp_in[i])
        RP_cvar_in[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[displ:end], w_cvar_in[i])
        RP_maxdiv_in[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[displ:end], w_maxdiv_in[i])
        displ = displ + k
        end = end + k

    listapesi_in = [w_meanvar_in, w_minvar_in, w_ew_in, w_rp_in, w_cvar_in, w_maxdiv_in]
    lista_RP_in = [RP_meanvar_in, RP_minvar_in, RP_ew_in, RP_rp_in, RP_cvar_in, RP_maxdiv_in]

    cumRP_meanvar_in = rendimenticumulati(RP_meanvar_in)
    cumRP_minvar_in = rendimenticumulati(RP_minvar_in)
    cumRP_ew_in = rendimenticumulati(RP_ew_in)
    cumRP_rp_in = rendimenticumulati(RP_rp_in)
    cumRP_cvar_in = rendimenticumulati(RP_cvar_in)
    cumRP_maxdiv_in = rendimenticumulati(RP_maxdiv_in)
    lista_cumRP_in = [cumRP_meanvar_in, cumRP_minvar_in, cumRP_ew_in, cumRP_rp_in, cumRP_cvar_in, cumRP_maxdiv_in]

    return dates, listapesi_in, lista_RP_in, lista_cumRP_in

# ----------------------------------------------------------------------------------------------------
# OUT-OF-SAMPLE PROCEDURE
def outofsample_approach (rendimenti_df,displ=100,k=7):
    i = 0
    from_ = 0
    steps = math.floor((len(rendimenti_df[:-1]) - displ) / k)
    n_asset = rendimenti_df.shape[1]
    l = k * steps
    dates=rendimenti_df.index[displ:displ+steps*k]
    w_meanvar, w_minvar, w_ew, w_rp, w_cvar, w_maxdiv = np.empty([steps, n_asset]), np.empty(
        [steps, n_asset]), np.empty([steps, n_asset]), np.empty([steps, n_asset]), np.empty(
        [steps, n_asset]), np.empty([steps, n_asset])
    RP_meanvar_oos, RP_minvar_oos, RP_ew_oos, RP_rp_oos, RP_cvar_oos, RP_maxdiv_oos = np.empty([l, ]), np.empty(
        [l, ]), np.empty([l, ]), np.empty([l, ]), np.empty([l, ]), np.empty([l, ])

    for i in range(steps):
        w_meanvar[i], w_minvar[i], w_ew[i], w_rp[i], w_cvar[i], w_maxdiv[i] = pesiportafogli(
            rendimenti_df[k * i: displ + k * i])
        from_ = displ + i * k
        RP_meanvar_oos[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[from_:from_ + k], w_meanvar[i])
        RP_minvar_oos[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[from_:from_ + k], w_minvar[i])
        RP_ew_oos[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[from_:from_ + k], w_ew[i])
        RP_rp_oos[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[from_:from_ + k], w_rp[i])
        RP_cvar_oos[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[from_:from_ + k], w_cvar[i])
        RP_maxdiv_oos[i * k:i * k + k] = rendimentiportafoglio(rendimenti_df[from_:from_ + k], w_maxdiv[i])


    lista_RP_oos = [RP_meanvar_oos, RP_minvar_oos, RP_ew_oos, RP_rp_oos, RP_cvar_oos, RP_maxdiv_oos]
    listapesi_oos = [w_meanvar, w_minvar, w_ew, w_rp, w_cvar, w_maxdiv]

    cumRP_meanvar_oos = rendimenticumulati(RP_meanvar_oos)
    cumRP_minvar_oos = rendimenticumulati(RP_minvar_oos)
    cumRP_ew_oos = rendimenticumulati(RP_ew_oos)
    cumRP_rp_oos = rendimenticumulati(RP_rp_oos)
    cumRP_cvar_oos = rendimenticumulati(RP_cvar_oos)
    cumRP_maxdiv_oos = rendimenticumulati(RP_maxdiv_oos)
    lista_cumRP_oos = [cumRP_meanvar_oos, cumRP_minvar_oos, cumRP_ew_oos, cumRP_rp_oos, cumRP_cvar_oos,
                       cumRP_maxdiv_oos]

    return dates, listapesi_oos, lista_RP_oos, lista_cumRP_oos


# ----------------------------------------------------------------------------------------------------------------------
# AVERAGE WEIGHTS TABLE
def pesimedi_sottoperiodi_green(lista_pesi_in, lista_pesi_oos, start_in, end_in, start_oos, end_oos):
    lista_copy_in=lista_pesi_in.copy()
    elem = 0
    for elem in range(len(lista_pesi_in)):
        w_in = lista_pesi_in[elem]
        w_in = w_in[start_in:end_in]
        lista_copy_in[elem] = w_in

    elem_oos = 0
    lista_copy_oos=lista_pesi_oos.copy()
    for elem_oos in range(len(lista_pesi_oos)):
        w_oos = lista_pesi_oos[elem_oos]
        w_oos = w_oos[start_oos:end_oos]
        lista_copy_oos[elem_oos] = w_oos

    composizione_media = composizione_media_portafogligreen(lista_copy_in, lista_copy_oos)

    return composizione_media

def pesimedi_sottoperiodi_nogreen(lista_pesi_in, lista_pesi_oos, start_in, end_in, start_oos, end_oos):
    lista_copy_in=lista_pesi_in.copy()
    elem = 0
    for strategy in range(len(lista_pesi_in)):
        w_in = lista_pesi_in[strategy]
        w_in = w_in[start_in:end_in]
        lista_copy_in[elem] = w_in

    elem_oos = 0
    lista_copy_oos=lista_pesi_oos.copy()
    for strategy_oos in range(len(lista_pesi_oos)):
        w_oos = lista_pesi_oos[strategy_oos]
        w_oos = w_oos[start_oos:end_oos]
        lista_copy_oos[strategy_oos] = w_oos

    composizione_media = composizione_media_portafogli(lista_copy_in, lista_copy_oos)

    return composizione_media

def tabellapesimedi(pesimedi_meanvar, pesimedi_minvar, pesimedi_ew, pesimedi_rp, pesimedi_cvar, pesimedi_maxdiv):
    assetclass = ['BBGB', 'SOLGB', 'BBBOND', 'MSCI','SPGSEN','SP5IAIR','SP5EHCR','SPEUIT']
    DF = pd.DataFrame(index = assetclass)
    DF['Portafoglio Mean Variance'] = pesimedi_meanvar
    DF['Portafoglio Minimum Variance'] = pesimedi_minvar
    DF['Portafoglio Equal Weight'] = pesimedi_ew
    DF['Portafoglio Risk Parity'] = pesimedi_rp
    DF['Portafoglio CVaR Optimization'] = pesimedi_cvar
    DF['Portafoglio Max Diversification'] = pesimedi_maxdiv
    return DF

def composizione_media_portafogligreen(lista_pesi_in, lista_pesi_oos):
    w1_in, w2_in, w3_in, w4_in, w5_in, w6_in = lista_pesi_in[0], lista_pesi_in[1], lista_pesi_in[2], lista_pesi_in[3], lista_pesi_in[4], lista_pesi_in[5]
    w1_oos, w2_oos, w3_oos, w4_oos, w5_oos, w6_oos = lista_pesi_oos[0], lista_pesi_oos[1], lista_pesi_oos[2], lista_pesi_oos[3], \
                                               lista_pesi_oos[4], lista_pesi_oos[5]
    tabellapesi_meanvar_in = pd.DataFrame(w1_in,
                                          columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                   'SPEUIT'])
    tabellapesi_minvar_in = pd.DataFrame(w2_in,
                                         columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                  'SPEUIT'])
    tabellapesi_ew_in = pd.DataFrame(w3_in,
                                     columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                              'SPEUIT'])
    tabellapesi_rp_in = pd.DataFrame(w4_in,
                                     columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                              'SPEUIT'])
    tabellapesi_cvar_in = pd.DataFrame(w5_in,
                                       columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                'SPEUIT'])
    tabellapesi_maxdiv_in = pd.DataFrame(w6_in,
                                         columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                  'SPEUIT'])

    # Out-of-sample
    tabellapesi_meanvar_oos = pd.DataFrame(w1_oos,
                                           columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                    'SPEUIT'])
    tabellapesi_minvar_oos = pd.DataFrame(w2_oos,
                                          columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                   'SPEUIT'])
    tabellapesi_ew_oos = pd.DataFrame(w3_oos, columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                     'SPEUIT'])
    tabellapesi_rp_oos = pd.DataFrame(w4_oos, columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                     'SPEUIT'])
    tabellapesi_cvar_oos = pd.DataFrame(w5_oos,
                                        columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                 'SPEUIT'])
    tabellapesi_maxdiv_oos = pd.DataFrame(w6_oos,
                                          columns=['BBGB', 'SOLGB', 'BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                   'SPEUIT'])
    pesimedi_meanvar_in = round(tabellapesi_meanvar_in.mean(axis=0) * 100, 1)
    pesimedi_minvar_in = round(tabellapesi_minvar_in.mean(axis=0) * 100, 1)
    pesimedi_ew_in = round(tabellapesi_ew_in.mean(axis=0) * 100, 1)
    pesimedi_rp_in = round(tabellapesi_rp_in.mean(axis=0) * 100, 1)
    pesimedi_cvar_in = round(tabellapesi_cvar_in.mean(axis=0) * 100, 1)
    pesimedi_maxdiv_in = round(tabellapesi_maxdiv_in.mean(axis=0) * 100, 1)

    pesimedi_meanvar_oos = round(tabellapesi_meanvar_oos.mean(axis=0) * 100, 1)
    pesimedi_minvar_oos = round(tabellapesi_minvar_oos.mean(axis=0) * 100, 1)
    pesimedi_ew_oos = round(tabellapesi_ew_oos.mean(axis=0) * 100, 1)
    pesimedi_rp_oos = round(tabellapesi_rp_oos.mean(axis=0) * 100, 1)
    pesimedi_cvar_oos = round(tabellapesi_cvar_oos.mean(axis=0) * 100, 1)
    pesimedi_maxdiv_oos = round(tabellapesi_maxdiv_oos.mean(axis=0) * 100, 1)

    # Tabella composizione media portafogli ** funzione solo per green
    composizioneportafogli_in = tabellapesimedi(pesimedi_meanvar_in, pesimedi_minvar_in, pesimedi_ew_in, pesimedi_rp_in,
                          pesimedi_cvar_in, pesimedi_maxdiv_in)
    composizioneportafogli_oos = tabellapesimedi(pesimedi_meanvar_oos, pesimedi_minvar_oos, pesimedi_ew_oos, pesimedi_rp_oos,
                          pesimedi_cvar_oos, pesimedi_maxdiv_oos)
    dfs = [composizioneportafogli_in, composizioneportafogli_oos]

    composizioneportafogli = pd.concat(dfs)

    return composizioneportafogli

def composizione_media_portafogli(lista_pesi_in, lista_pesi_oos):
    w1_in, w2_in, w3_in, w4_in, w5_in, w6_in = lista_pesi_in[0], lista_pesi_in[1], lista_pesi_in[2], lista_pesi_in[3], \
                                               lista_pesi_in[4], lista_pesi_in[5]
    w1_oos, w2_oos, w3_oos, w4_oos, w5_oos, w6_oos = lista_pesi_oos[0], lista_pesi_oos[1], lista_pesi_oos[2], \
                                                     lista_pesi_oos[3], \
                                                     lista_pesi_oos[4], lista_pesi_oos[5]
    # In-sample
    tabellapesi_meanvar_in = pd.DataFrame(w1_in,
                                          columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                   'SPEUIT'])
    tabellapesi_minvar_in = pd.DataFrame(w2_in,
                                         columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                  'SPEUIT'])
    tabellapesi_ew_in = pd.DataFrame(w3_in,
                                     columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                              'SPEUIT'])
    tabellapesi_rp_in = pd.DataFrame(w4_in,
                                     columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                              'SPEUIT'])
    tabellapesi_cvar_in = pd.DataFrame(w5_in,
                                       columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                'SPEUIT'])
    tabellapesi_maxdiv_in = pd.DataFrame(w6_in,
                                         columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                  'SPEUIT'])

    # Out-of-sample
    tabellapesi_meanvar_oos = pd.DataFrame(w1_oos,
                                           columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                    'SPEUIT'])
    tabellapesi_minvar_oos = pd.DataFrame(w2_oos,
                                          columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                   'SPEUIT'])
    tabellapesi_ew_oos = pd.DataFrame(w3_oos,
                                      columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                               'SPEUIT'])
    tabellapesi_rp_oos = pd.DataFrame(w4_oos,
                                      columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                               'SPEUIT'])
    tabellapesi_cvar_oos = pd.DataFrame(w5_oos,
                                        columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                 'SPEUIT'])
    tabellapesi_maxdiv_oos = pd.DataFrame(w6_oos,
                                          columns=['BBBOND', 'MSCI', 'SPGSEN', 'SP5IAIR', 'SP5EHCR',
                                                   'SPEUIT'])
    pesimedi_meanvar_in = round(tabellapesi_meanvar_in.mean(axis=0) * 100, 1)
    pesimedi_minvar_in = round(tabellapesi_minvar_in.mean(axis=0) * 100, 1)
    pesimedi_ew_in = round(tabellapesi_ew_in.mean(axis=0) * 100, 1)
    pesimedi_rp_in = round(tabellapesi_rp_in.mean(axis=0) * 100, 1)
    pesimedi_cvar_in = round(tabellapesi_cvar_in.mean(axis=0) * 100, 1)
    pesimedi_maxdiv_in = round(tabellapesi_maxdiv_in.mean(axis=0) * 100, 1)

    pesimedi_meanvar_oos = round(tabellapesi_meanvar_oos.mean(axis=0) * 100, 1)
    pesimedi_minvar_oos = round(tabellapesi_minvar_oos.mean(axis=0) * 100, 1)
    pesimedi_ew_oos = round(tabellapesi_ew_oos.mean(axis=0) * 100, 1)
    pesimedi_rp_oos = round(tabellapesi_rp_oos.mean(axis=0) * 100, 1)
    pesimedi_cvar_oos = round(tabellapesi_cvar_oos.mean(axis=0) * 100, 1)
    pesimedi_maxdiv_oos = round(tabellapesi_maxdiv_oos.mean(axis=0) * 100, 1)

    # Average portfolio weights table ** only for green portfolios
    composizioneportafogli_in = tabellapesimedi(pesimedi_meanvar_in, pesimedi_minvar_in, pesimedi_ew_in, pesimedi_rp_in,
                                                pesimedi_cvar_in, pesimedi_maxdiv_in)
    composizioneportafogli_oos = tabellapesimedi(pesimedi_meanvar_oos, pesimedi_minvar_oos, pesimedi_ew_oos,
                                                 pesimedi_rp_oos,
                                                 pesimedi_cvar_oos, pesimedi_maxdiv_oos)
    dfs = [composizioneportafogli_in, composizioneportafogli_oos]

    composizioneportafogli = pd.concat(dfs)

    return composizioneportafogli
