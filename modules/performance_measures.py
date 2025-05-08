import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import empyrical as em
import scipy
from scipy.signal import find_peaks

plt.rcParams['figure.dpi'] = 300    ###### increase image resolution
plt.rcParams['savefig.dpi'] = 300   ######

# PERFORMANCE MEASURE TABLE
def tabella(rp_meanvar, rp_minvar, rp_ew, rp_rp, rp_cvar, rp_maxdiv):
    misureperf_l = ['Annualized return', 'Annualized volatility', 'Sharpe ratio', 'Downside risk',
                    'Omega ratio','Maximum Drawdown', 'VaR 5%', 'CVaR 5%']
    df = pd.DataFrame(index = misureperf_l)
    df['Portafoglio Mean Variance'] = 0
    df['Portafoglio Minimum Variance'] = 0
    df['Portafoglio Equal Weight'] = 0
    df['Portafoglio Risk Parity'] = 0
    df['Portafoglio CVaR Optimization'] = 0
    df['Portafoglio Max. Diversification'] = 0
    # lista portafogli inseriti in input
    lista_portafogli = [rp_meanvar, rp_minvar, rp_ew, rp_rp, rp_cvar, rp_maxdiv]

    i = 0
    for portafoglio in lista_portafogli:
        misperf_ar = np.empty([8,1])
        misperf_ar[0] = em.annual_return(portafoglio) * 100
        misperf_ar[1] = em.annual_volatility(portafoglio) * 100
        misperf_ar[2] = em.sharpe_ratio(portafoglio)
        misperf_ar[3] = em.downside_risk(portafoglio) * 100
        misperf_ar[4] = em.omega_ratio(portafoglio)
        misperf_ar[5] = em.max_drawdown(portafoglio) * 100
        misperf_ar[6] = em.value_at_risk(portafoglio, 0.05) * 100
        misperf_ar[7] = ExpectedShortfall5(portafoglio) * 100
        df.iloc[:,i] = misperf_ar
        i = i + 1

    return df


# EXPECTED SHORTFALL
def ExpectedShortfall5(rendimentiportafoglio):
    var = np.percentile(rendimentiportafoglio, 5)
    return rendimentiportafoglio[rendimentiportafoglio <= var].mean()

###########adding dependence from displ
def analisi_sottoperiodo (rendimenti_df, listaRP_in, lista_RP_oos, listaCumRP_in, listaCumRP_oos, start1, end1,folder, period, displ=100):
    RP_meanvar_in = listaRP_in[0]
    RP_minvar_in = listaRP_in[1]
    RP_ew_in = listaRP_in[2]
    RP_rp_in = listaRP_in[3]
    RP_cvar_in = listaRP_in[4]
    RP_maxdiv_in = listaRP_in[5]

    RP_meanvar_in_SubP = RP_meanvar_in[start1:end1].copy()
    RP_minvar_in_SubP = RP_minvar_in[start1:end1].copy()
    RP_ew_in_SubP = RP_ew_in[start1:end1].copy()
    RP_rp_in_SubP = RP_rp_in[start1:end1].copy()
    RP_cvar_in_SubP = RP_cvar_in[start1:end1].copy()
    RP_maxdiv_in_SubP = RP_maxdiv_in[start1:end1].copy()

    RP_meanvar_oos = lista_RP_oos[0]
    RP_minvar_oos = lista_RP_oos[1]
    RP_ew_oos = lista_RP_oos[2]
    RP_rp_oos = lista_RP_oos[3]
    RP_cvar_oos = lista_RP_oos[4]
    RP_maxdiv_oos = lista_RP_oos[5]

    RP_meanvar_oos_SubP = RP_meanvar_oos[start1:end1].copy()
    RP_minvar_oos_SubP = RP_minvar_oos[start1:end1].copy()
    RP_ew_oos_SubP = RP_ew_oos[start1:end1].copy()
    RP_rp_oos_SubP = RP_rp_oos[start1:end1].copy()
    RP_cvar_oos_SubP = RP_cvar_oos[start1:end1].copy()
    RP_maxdiv_oos_SubP = RP_maxdiv_oos[start1:end1].copy()

    performance_insample = tabella(RP_meanvar_in_SubP, RP_minvar_in_SubP, RP_ew_in_SubP, RP_rp_in_SubP,
                                           RP_cvar_in_SubP, RP_maxdiv_in_SubP)

    performance_outofsample = tabella(RP_meanvar_oos_SubP, RP_minvar_oos_SubP, RP_ew_oos_SubP, RP_rp_oos_SubP,
                                              RP_cvar_oos_SubP, RP_maxdiv_oos_SubP)

    # # In-sample cumulated returns
    # cumRP_meanvar_in = listaCumRP_in[0]
    # cumRP_minvar_in = listaCumRP_in[1]
    # cumRP_ew_in = listaCumRP_in[2]
    # cumRP_rp_in = listaCumRP_in[3]
    # cumRP_cvar_in = listaCumRP_in[4]
    # cumRP_maxdiv_in = listaCumRP_in[5]

    # # Out-of-sample cumulated returns
    # cumRP_meanvar_oos = listaCumRP_oos[0]
    # cumRP_minvar_oos = listaCumRP_oos[1]
    # cumRP_ew_oos = listaCumRP_oos[2]
    # cumRP_rp_oos = listaCumRP_oos[3]
    # cumRP_cvar_oos = listaCumRP_oos[4]
    # cumRP_maxdiv_oos = listaCumRP_oos[5]

    # grafico = plt.figure(figsize=(20, 10))
    # graficoinsample = plt.subplot(1, 2, 1)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_meanvar_in[start1:end1], 'k-', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_minvar_in[start1:end1], 'k--', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_ew_in[start1:end1],color='#925591',linestyle='dashed', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_rp_in[start1:end1],color='#925591',linestyle='dotted', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_cvar_in[start1:end1],'k-.', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_maxdiv_in[start1:end1],color='#925591',linestyle='dashdot', linewidth=0.8)
    # # plt.legend(['Mean Variance', 'Minimum Variance', '1/N', 'Risk Parity', 'CVar opt', 'Max Div'])
    # # #plt.title('In-sample Portfolios', fontsize = 20)
    # # minimo = min(min(cumRP_meanvar_in[start1:end1]), min(cumRP_minvar_in[start1:end1]), min(cumRP_ew_in[start1:end1]),
    # #           min(cumRP_rp_in[start1:end1]), min(cumRP_cvar_in[start1:end1]), min(cumRP_maxdiv_in[start1:end1]),
    # #           min(cumRP_meanvar_oos[start1:end1]), min(cumRP_minvar_oos[start1:end1]), min(cumRP_ew_oos[start1:end1]),
    # #           min(cumRP_rp_oos[start1:end1]), min(cumRP_cvar_oos[start1:end1]), min(cumRP_maxdiv_oos[start1:end1]))
    # # massimo = max(max(cumRP_meanvar_in[start1:end1]), max(cumRP_minvar_in[start1:end1]), max(cumRP_ew_in[start1:end1]),
    # #           max(cumRP_rp_in[start1:end1]), max(cumRP_cvar_in[start1:end1]), max(cumRP_maxdiv_in[start1:end1]),
    # #           max(cumRP_meanvar_oos[start1:end1]), max(cumRP_minvar_oos[start1:end1]), max(cumRP_ew_oos[start1:end1]),
    # #           max(cumRP_rp_oos[start1:end1]), max(cumRP_cvar_oos[start1:end1]), max(cumRP_maxdiv_oos[start1:end1]),)
    # # plt.ylim((minimo-0.1), (massimo+0.1))
    # # graficooutofsample = plt.subplot(1, 2, 2)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_meanvar_oos[start1:end1],'k-', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_minvar_oos[start1:end1],'k--', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_ew_oos[start1:end1],color='#925591',linestyle='dashed', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_rp_oos[start1:end1],color='#925591',linestyle='dotted', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_cvar_oos[start1:end1],'k-.', linewidth=0.8)
    # # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_maxdiv_oos[start1:end1],color='#925591',linestyle='dashdot', linewidth=0.8)
    # # plt.legend(['Mean Variance', 'Minimum Variance', '1/N', 'Risk Parity', 'CVar opt', 'Max Div'])
    # # #plt.title('Out-of-sample Portfolios', fontsize = 20)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_meanvar_in[start1:end1],linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_minvar_in[start1:end1],linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_ew_in[start1:end1], linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_rp_in[start1:end1], linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_cvar_in[start1:end1],linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_maxdiv_in[start1:end1], linewidth=0.8)
    # plt.xticks(fontsize = 15, rotation = 45)
    # plt.yticks(fontsize = 15)
    # plt.legend(['Mean Variance', 'Minimum Variance', '1/N', 'Risk Parity', 'CVar opt', 'Max Div'], fontsize=15)
    # #plt.title('In-sample Portfolios', fontsize = 20)
    # minimo = min(min(cumRP_meanvar_in[start1:end1]), min(cumRP_minvar_in[start1:end1]), min(cumRP_ew_in[start1:end1]),
    #           min(cumRP_rp_in[start1:end1]), min(cumRP_cvar_in[start1:end1]), min(cumRP_maxdiv_in[start1:end1]),
    #           min(cumRP_meanvar_oos[start1:end1]), min(cumRP_minvar_oos[start1:end1]), min(cumRP_ew_oos[start1:end1]),
    #           min(cumRP_rp_oos[start1:end1]), min(cumRP_cvar_oos[start1:end1]), min(cumRP_maxdiv_oos[start1:end1]))
    # massimo = max(max(cumRP_meanvar_in[start1:end1]), max(cumRP_minvar_in[start1:end1]), max(cumRP_ew_in[start1:end1]),
    #           max(cumRP_rp_in[start1:end1]), max(cumRP_cvar_in[start1:end1]), max(cumRP_maxdiv_in[start1:end1]),
    #           max(cumRP_meanvar_oos[start1:end1]), max(cumRP_minvar_oos[start1:end1]), max(cumRP_ew_oos[start1:end1]),
    #           max(cumRP_rp_oos[start1:end1]), max(cumRP_cvar_oos[start1:end1]), max(cumRP_maxdiv_oos[start1:end1]),)
    # plt.ylim((minimo-0.1), (massimo+0.1))
    # graficooutofsample = plt.subplot(1, 2, 2)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_meanvar_oos[start1:end1],linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_minvar_oos[start1:end1],linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_ew_oos[start1:end1],linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_rp_oos[start1:end1],linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_cvar_oos[start1:end1],linewidth=0.8)
    # plt.plot(rendimenti_df[(start1+displ):(end1+displ)].index, cumRP_maxdiv_oos[start1:end1], linewidth=0.8)
    # plt.xticks(fontsize = 15, rotation = 45)
    # plt.yticks(fontsize = 15)
    # plt.legend(['Mean Variance', 'Minimum Variance', '1/N', 'Risk Parity', 'CVar opt', 'Max Div'], fontsize=15)
    # #plt.title('Out-of-sample Portfolios', fontsize = 20)
    # plt.ylim((minimo-0.1), (massimo+0.1))
    # plt.savefig('{}/{}.png'.format(folder,period))
    # plt.show()
    
    return performance_insample, performance_outofsample

def windows(data,plot=False):
    cd=-1*data
    peaks, _ = find_peaks(cd,prominence=(0.04, None))
    #The prominence of a peak measures how much a peak stands out from the 
    #surrounding baseline of the signal and is defined as the vertical distance 
    #between the peak and its lowest contour line
    if plot == True:
        plt.plot(data)
        plt.plot(peaks, data[peaks], "x")
        plt.plot(np.zeros_like(data), "--", color="gray")
        plt.show()
    w=len(data)/len(peaks)
    return np.ceil(w)