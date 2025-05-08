import pandas as pd
import numpy as np
import datetime
import math
import modules.strategies as s
import modules.performance_measures as mp
import os
from modules.spectral_risk_measure import spectral_measure
from modules.spectral_risk_measure import quantile
from modules.spectral_risk_measure import plot_SRM
from modules.performance_measures import windows

########################GREEN PORTFOLIOS###############################################

# IMPORT PRICES
directory_path = os.getcwd()
prezzi_df = pd.read_excel(directory_path + '\\input_data.xlsx')
prezzi_df = prezzi_df.fillna(method='ffill')
prezzi_df = prezzi_df.set_index(prezzi_df.Date)
del prezzi_df['Date']
# TAKE LOG RETURNS
cols=prezzi_df.columns
index_list = prezzi_df.index
log_returns=np.diff(np.log(prezzi_df),axis=0)
rendimenti_df=pd.DataFrame(log_returns, columns=cols, index=index_list[1:]) #remove first date, it disappears when taking log diff
rendimenti_df=rendimenti_df.dropna()
# SET PARAMETERS
#choice of rolling window length based on shock persistence
c=np.where(prezzi_df.columns=='MSCI')[0][0] #find column index of stock market
window = windows((log_returns[:,c]+1.0).cumprod(), plot=False) #find average days between shocks
displ = int(window) #length of rolling window is the rounded value
#-----------
k_in=len(rendimenti_df)-1-displ #in-sample recalibration periodicity 
k_oos=7  #out-of-sample recalibration periodicity 
plot_folder='plots/green_portfolios' #plot output folder

# -------------------------IN-SAMPLE PROCEDURE-------------------------------------
dates_in, listapesi_in, listaRP_in, lista_cumRP_in = s.insample_approach(rendimenti_df, displ,k_in)

# -------------------------OUT OF SAMPLE PROCEDURE---------------------------------
dates_oos, listapesi_oos, listaRP_oos, lista_cumRP_oos = s.outofsample_approach(rendimenti_df, displ,k_oos)

subperiods= { #american date format
  "bear": ['05/21/2015','04/11/2016'],
  "bull": ['12/01/2016','01/31/2018'],
  "pre-pand": ['01/03/2019','03/03/2020'],
  "pand": ['03/04/2020','06/07/2021']
}
subperiod_names=[key for key in subperiods]
#We create the indices corresponding to each date (shifted by displ)
j=0
for i in subperiods:    
    if i != subperiod_names[-1]:
        subperiods[i]=[int(np.where(dates_in==datetime.datetime.strptime(subperiods[i][0], '%m/%d/%Y'))[0]),
                          int(np.where(dates_in==datetime.datetime.strptime(subperiods[i][1], '%m/%d/%Y'))[0])]
    else:
        subperiods[i]=[int(np.where(dates_in==datetime.datetime.strptime(subperiods[i][0], '%m/%d/%Y'))[0]),
                          min(len(dates_in)-1,len(dates_oos)-1)] #the last element depends on k, t's not necessarily the final observation
    j+=1


# PERFORMANCE
# In sample
performance_insample = mp.tabella(listaRP_in[0],listaRP_in[1], listaRP_in[2], listaRP_in[3], listaRP_in[4], listaRP_in[5])
# Out of sample
performance_outofsample = mp.tabella(listaRP_oos[0], listaRP_oos[1], listaRP_oos[2], listaRP_oos[3], listaRP_oos[4], listaRP_oos[5])

# By sub-period
# BEAR
performance_insample_BEAR, performance_outofsample_BEAR = mp.analisi_sottoperiodo(rendimenti_df, listaRP_in,
                                                                                  listaRP_oos, lista_cumRP_in, lista_cumRP_oos, subperiods[subperiod_names[0]][0], subperiods[subperiod_names[0]][1]+1,plot_folder,subperiod_names[0],displ)
# BULL
performance_insample_BULL, performance_outofsample_BULL = mp.analisi_sottoperiodo(rendimenti_df, listaRP_in,
                                                                                  listaRP_oos, lista_cumRP_in, lista_cumRP_oos, subperiods[subperiod_names[1]][0], subperiods[subperiod_names[1]][1]+1,plot_folder,subperiod_names[1],displ)
# PRE-PANDEMIC
performance_insample_PREPANDEMIC, performance_outofsample_PREPANDEMIC = mp.analisi_sottoperiodo(rendimenti_df, listaRP_in,
                                                                                  listaRP_oos, lista_cumRP_in, lista_cumRP_oos, subperiods[subperiod_names[2]][0], subperiods[subperiod_names[2]][1]+1,plot_folder,subperiod_names[2],displ)
# PANDEMIC  
performance_insample_PANDEMIC, performance_outofsample_PANDEMIC = mp.analisi_sottoperiodo(rendimenti_df, listaRP_in,
                                                                                  listaRP_oos, lista_cumRP_in, lista_cumRP_oos, subperiods[subperiod_names[3]][0], subperiods[subperiod_names[3]][1]+1,plot_folder,subperiod_names[3],displ)
#We find the indices in terms of calibration windows corresponding to each subperiod
indices_in=[[math.floor(subperiods[i][0]/k_in),math.floor(subperiods[i][1]/k_in)] for i in subperiods]
indices_oos=[ [round(subperiods[i][0]/k_oos),round(subperiods[i][1]/k_oos)] for i in subperiods]
indices_oos[-1][1]=round(len(dates_oos)/k_oos) #oos and in have different lengths beacause k is different
indices_in[-1][1]=math.floor((len(dates_in)-1)/k_in)

# PORTFOLIO AVERAGE WEIGHTS
composizioneportafogli = s.composizione_media_portafogligreen(listapesi_in, listapesi_oos)
# BEAR
composizioneportafogli_BEAR = s.pesimedi_sottoperiodi_green(listapesi_in, listapesi_oos, indices_in[0][0], indices_in[0][1]+1, indices_oos[0][0], indices_oos[0][1])
# BULL
composizioneportafogli_BULL = s.pesimedi_sottoperiodi_green(listapesi_in, listapesi_oos, indices_in[1][0], indices_in[1][1]+1, indices_oos[1][0], indices_oos[1][1])
# PREPANDEMIC
composizioneportafogli_PREPANDEMIC = s.pesimedi_sottoperiodi_green(listapesi_in, listapesi_oos, indices_in[2][0], indices_in[2][1]+1, indices_oos[2][0], indices_oos[2][1])
# PREPANDEMIC
composizioneportafogli_PANDEMIC = s.pesimedi_sottoperiodi_green(listapesi_in, listapesi_oos, indices_in[3][0], indices_in[3][1]+1, indices_oos[3][0], indices_oos[3][1])

# EXPORT
with pd.ExcelWriter('excel_tables/partialtables/'+"asset_allocation_greenoutput.xlsx") as writer:
    composizioneportafogli.to_excel(writer, sheet_name="Portafogli")
    performance_insample.to_excel(writer, sheet_name='Performance_IN')
    performance_outofsample.to_excel(writer, sheet_name='Performance_OOS')
    performance_insample_BEAR.to_excel(writer, sheet_name='BEAR_Performance_IN')
    performance_outofsample_BEAR.to_excel(writer, sheet_name='BEAR_Performance_OOS')
    performance_insample_BULL.to_excel(writer, sheet_name='BULL_Performance_IN')
    performance_outofsample_BULL.to_excel(writer, sheet_name='BULL_Performance_OOS')
    performance_insample_PREPANDEMIC.to_excel(writer, sheet_name='PREPANDEMIC_Performance_IN')
    performance_outofsample_PREPANDEMIC.to_excel(writer, sheet_name='PREPANDEMIC_Performance_OOS')
    performance_insample_PANDEMIC.to_excel(writer, sheet_name='PANDEMIC_Performance_IN')
    performance_outofsample_PANDEMIC.to_excel(writer, sheet_name='PANDEMIC_Performance_OOS')
    composizioneportafogli_BEAR.to_excel(writer, sheet_name='Portafogli_BEAR')
    composizioneportafogli_BULL.to_excel(writer, sheet_name='Portafogli_BULL')
    composizioneportafogli_PREPANDEMIC.to_excel(writer, sheet_name='Portafogli_PREPANDEMIC')
    composizioneportafogli_PANDEMIC.to_excel(writer, sheet_name='Portafogli_PANDEMIC')

# SPECTRAL ANALYSIS
# Parameters
ns=len(listaRP_in) #number of allocation strategies
nper=5 #number of periods
steps=1000  #empirical function step size
k_list=[10,20,30,50] #risk-aversion parameters
#add 'all' period to dictionary
subperiod_names.insert(0,'all')
subperiods['all']=[0,subperiods[subperiod_names[-1]][1]]

# Empty dictionaries to contain quantile and SRM results:
# For quantiles
quantile_dict_in={} #out of sample quantiles
quantile_dict_oos={} #out of sample quantiles
for i in subperiod_names:
    quantile_dict_in[i]=[]
    quantile_dict_oos[i]=[] #out of sample

for key in quantile_dict_in: #Adding as many empty lists per period as there are allocation strategies
    for strategy in range(0,ns):
        quantile_dict_in[key].append([])
        quantile_dict_oos[key].append([])
# For SRM results
subperiod_names=[w.title() for w in subperiod_names]
spectral_dict_green_in={}
spectral_dict_green_oos={}
spectral_df_in=pd.DataFrame(np.zeros((ns,nper)),columns=subperiod_names)
spectral_df_oos=pd.DataFrame(np.zeros((ns,nper)),columns=subperiod_names)
# Find empirical quantiles per period and per strategy
# Find SRM per period i, per strategy, per risk-aversion parameter k
for k in k_list:
    for i in subperiods:
        start=subperiods[i][0]
        stop=subperiods[i][1]
        for strategy in range(0,ns):    
            quantile_dict_in[i][strategy]=quantile(listaRP_in[strategy][start:stop],steps)
            quantile_dict_oos[i][strategy]=quantile(listaRP_oos[strategy][start:stop],steps)
            spectral_df_in.loc[strategy,i.title()]=float(spectral_measure(quantile_dict_in[i][strategy],k,steps))
            spectral_df_oos.loc[strategy,i.title()]=float(spectral_measure(quantile_dict_oos[i][strategy],k,steps))  
    spectral_dict_green_in[k]=spectral_df_in
    spectral_dict_green_oos[k]=spectral_df_oos
    spectral_df_in=pd.DataFrame(np.zeros((ns,nper)),columns=subperiod_names) #empty out for new loop
    spectral_df_oos=pd.DataFrame(np.zeros((ns,nper)),columns=subperiod_names) #empty out for new loop


# EXPORT TO EXCEL FILE for GREEN PORTFOLIOS
with pd.ExcelWriter('excel_tables/partialtables/'+"Spectral_output_GREEN.xlsx") as writer:
    for k in k_list:
        spectral_dict_green_in[k].to_excel(writer, sheet_name="spectral_in %s" % k)
        spectral_dict_green_oos[k].to_excel(writer, sheet_name="spectral_oos %s" % k)









########################NON-GREEN PORTFOLIOS###############################################

#IMPORT PRICES
directory_path = os.getcwd()
prezzi_df = pd.read_excel(directory_path + '\\input_data.xlsx')
prezzi_df = prezzi_df.fillna(method='ffill')
prezzi_df = prezzi_df.set_index(prezzi_df.Date)
del prezzi_df['Date'], prezzi_df['BBGB'], prezzi_df['SOLGB']
#TAKE LOG RETURNS
cols=prezzi_df.columns
index_list = prezzi_df.index
log_returns=np.diff(np.log(prezzi_df),axis=0)
rendimenti_df=pd.DataFrame(log_returns, columns=cols, index=index_list[1:]) #remove first date, it disappears when taking log diff
rendimenti_df=rendimenti_df.dropna()
#SET PARAMETERS
displ = 50 #length of rolling window
k_in=len(rendimenti_df)-1-displ #in-sample recalibration periodicity 
k_oos=7  #out-of-sample recalibration periodicity
plot_folder='plots/nongreen_portfolios' #plot output folder

# -------------------------IN-SAMPLE PROCEDURE-----------------------------------------
dates_in, listapesi_in, listaRP_in, lista_cumRP_in = s.insample_approach(rendimenti_df, displ, k_in)

# -------------------------OUT OF SAMPLE PROCEDURE---------------------------------
dates_oos, listapesi_oos, listaRP_oos, lista_cumRP_oos = s.outofsample_approach(rendimenti_df, displ, k_oos)


listapesi_copy_in, listapesi_copy_oos = listapesi_in, listapesi_oos
listaRP_copy_in, lista_cumRP_copy_in, listaRP_copy_oos, lista_cumRP_copy_oos  =listaRP_in, lista_cumRP_in, listaRP_oos, lista_cumRP_oos 

subperiods= { #american date format
  "bear": ['05/21/2015','04/11/2016'],
  "bull": ['12/01/2016','01/31/2018'],
  "pre-pand": ['01/03/2019','03/03/2020'],
  "pand": ['03/04/2020','06/07/2021']
}
subperiod_names=[key for key in subperiods]
#We create the indices corresponding to each date (shifted by displ)
j=0
for i in subperiods:    
    if i != subperiod_names[-1]:
        subperiods[i]=[int(np.where(dates_in==datetime.datetime.strptime(subperiods[i][0], '%m/%d/%Y'))[0]),
                          int(np.where(dates_in==datetime.datetime.strptime(subperiods[i][1], '%m/%d/%Y'))[0])]
    else:
        subperiods[i]=[int(np.where(dates_in==datetime.datetime.strptime(subperiods[i][0], '%m/%d/%Y'))[0]),
                          min(len(dates_in)-1,len(dates_oos)-1)] #the last element depends on k, t's not necessarily the final observation
    j+=1
    
# PERFORMANCE
# In sample
performance_insample = mp.tabella(listaRP_in[0],listaRP_in[1], listaRP_in[2], listaRP_in[3], listaRP_in[4], listaRP_in[5])
# Out of sample
performance_outofsample = mp.tabella(listaRP_oos[0], listaRP_oos[1], listaRP_oos[2], listaRP_oos[3], listaRP_oos[4], listaRP_oos[5])

# By sub-period
# BEAR
performance_insample_BEAR, performance_outofsample_BEAR = mp.analisi_sottoperiodo(rendimenti_df, listaRP_in,
                                                                                  listaRP_oos, lista_cumRP_in, lista_cumRP_oos, subperiods[subperiod_names[0]][0], subperiods[subperiod_names[0]][1]+1,plot_folder,subperiod_names[0],displ)
# BULL
performance_insample_BULL, performance_outofsample_BULL = mp.analisi_sottoperiodo(rendimenti_df, listaRP_in,
                                                                                  listaRP_oos, lista_cumRP_in, lista_cumRP_oos, subperiods[subperiod_names[1]][0], subperiods[subperiod_names[1]][1]+1,plot_folder,subperiod_names[1],displ)
# PRE-PANDEMIC
performance_insample_PREPANDEMIC, performance_outofsample_PREPANDEMIC = mp.analisi_sottoperiodo(rendimenti_df, listaRP_in,
                                                                                  listaRP_oos, lista_cumRP_in, lista_cumRP_oos, subperiods[subperiod_names[2]][0], subperiods[subperiod_names[2]][1]+1,plot_folder,subperiod_names[2],displ)
# PANDEMIC  
performance_insample_PANDEMIC, performance_outofsample_PANDEMIC = mp.analisi_sottoperiodo(rendimenti_df, listaRP_in,
                                                                                  listaRP_oos, lista_cumRP_in, lista_cumRP_oos, subperiods[subperiod_names[3]][0], subperiods[subperiod_names[3]][1]+1,plot_folder,subperiod_names[3],displ)
#We find the indices in terms of calibration windows corresponding to each subperiod
indices_in=[[round(subperiods[i][0]/k_in),round(subperiods[i][1]/k_in)] for i in subperiods]
indices_oos=[ [round(subperiods[i][0]/k_oos),round(subperiods[i][1]/k_oos)] for i in subperiods]
indices_oos[-1][1]=round(len(dates_oos)/k_oos) #oos and in have different lengths beacause k is different
indices_in[-1][1]=round(len(dates_in)/k_in)

# PORTFOLIO AVERAGE WEIGHTS
composizioneportafogli = s.composizione_media_portafogli(listapesi_in, listapesi_oos)
#BEAR
composizioneportafogli_BEAR = s.pesimedi_sottoperiodi_nogreen(listapesi_in, listapesi_oos, indices_in[0][0], indices_in[0][1]+1, indices_oos[0][0], indices_oos[0][1]) #####
#BULL
composizioneportafogli_BULL = s.pesimedi_sottoperiodi_nogreen(listapesi_in, listapesi_oos, indices_in[1][0], indices_in[1][1]+1, indices_oos[1][0], indices_oos[1][1])
#PREPANDEMIC
composizioneportafogli_PREPANDEMIC = s.pesimedi_sottoperiodi_nogreen(listapesi_in, listapesi_oos, indices_in[2][0]-1, indices_in[2][1]+1, indices_oos[2][0], indices_oos[2][1])
#PREPANDEMIC
composizioneportafogli_PANDEMIC = s.pesimedi_sottoperiodi_nogreen(listapesi_in, listapesi_oos, indices_in[3][0]-1, indices_in[3][1]+1, indices_oos[3][0], indices_oos[3][1])

#EXPORT
with pd.ExcelWriter('excel_tables/partialtables/'+"asset_allocation_nongreenoutput.xlsx") as writer:
    composizioneportafogli.to_excel(writer, sheet_name="Portafogli")
    performance_insample.to_excel(writer, sheet_name='Performance_IN')
    performance_outofsample.to_excel(writer, sheet_name='Performance_OOS')
    performance_insample_BEAR.to_excel(writer, sheet_name='BEAR_Performance_IN')
    performance_outofsample_BEAR.to_excel(writer, sheet_name='BEAR_Performance_OOS')
    performance_insample_BULL.to_excel(writer, sheet_name='BULL_Performance_IN')
    performance_outofsample_BULL.to_excel(writer, sheet_name='BULL_Performance_OOS')
    performance_insample_PREPANDEMIC.to_excel(writer, sheet_name='PREPANDEMIC_Performance_IN')
    performance_outofsample_PREPANDEMIC.to_excel(writer, sheet_name='PREPANDEMIC_Performance_OOS')
    performance_insample_PANDEMIC.to_excel(writer, sheet_name='PANDEMIC_Performance_IN')
    performance_outofsample_PANDEMIC.to_excel(writer, sheet_name='PANDEMIC_Performance_OOS')
    composizioneportafogli_BEAR.to_excel(writer, sheet_name='Portafogli_BEAR')
    composizioneportafogli_BULL.to_excel(writer, sheet_name='Portafogli_BULL')
    composizioneportafogli_PREPANDEMIC.to_excel(writer, sheet_name='Portafogli_PREPANDEMIC')
    composizioneportafogli_PANDEMIC.to_excel(writer, sheet_name='Portafogli_PANDEMIC')

# SPECTRAL ANALYSIS
# Parameters
ns=len(listaRP_in) #number of allocation strategies
nper=5 #number of periods
steps=1000  #empirical function step size
k_list=[10,20,30,50] #risk-aversion parameters
#add 'all' period to dictionary
subperiod_names.insert(0,'all')
subperiods['all']=[0,subperiods[subperiod_names[-1]][1]]

# Empty dictionaries to contain quantile and SRM results:
# For quantiles
quantile_dict_in={} #out of sample quantiles
quantile_dict_oos={} #out of sample quantiles
for i in subperiod_names:
    quantile_dict_in[i]=[]
    quantile_dict_oos[i]=[] #out of sample

for key in quantile_dict_in: #Adding as many empty lists per period as there are allocation strategies
    for strategy in range(0,ns):
        quantile_dict_in[key].append([])
        quantile_dict_oos[key].append([])
# For SRM results
subperiod_names=[w.title() for w in subperiod_names]
spectral_dict_nogreen_in={}
spectral_dict_nogreen_oos={}
spectral_df_in=pd.DataFrame(np.zeros((ns,nper)),columns=subperiod_names)
spectral_df_oos=pd.DataFrame(np.zeros((ns,nper)),columns=subperiod_names)
# Find empirical quantiles per period and per strategy
# Find SRM per period i, per strategy, per risk-aversion parameter k
for k in k_list:
    for i in subperiods:
        start=subperiods[i][0]
        stop=subperiods[i][1]
        for strategy in range(0,ns):
            quantile_dict_in[i][strategy]=quantile(listaRP_in[strategy][start:stop],steps)
            quantile_dict_oos[i][strategy]=quantile(listaRP_oos[strategy][start:stop],steps)
            spectral_df_in.loc[strategy,i.title()]=float(spectral_measure(quantile_dict_in[i][strategy],k,steps))
            spectral_df_oos.loc[strategy,i.title()]=float(spectral_measure(quantile_dict_oos[i][strategy],k,steps))  
    spectral_dict_nogreen_in[k]=spectral_df_in
    spectral_dict_nogreen_oos[k]=spectral_df_oos
    spectral_df_in=pd.DataFrame(np.zeros((ns,nper)),columns=subperiod_names) #empty out for new loop
    spectral_df_oos=pd.DataFrame(np.zeros((ns,nper)),columns=subperiod_names) #empty out for new loop

# EXPORT TO EXCEL FILE for NO GREEN portfolios
with pd.ExcelWriter('excel_tables/partialtables/'+"Spectral_output_NO_GREEN.xlsx") as writer:
    for k in k_list:
        spectral_dict_nogreen_in[k].to_excel(writer, sheet_name="spectral_in %s" % k)
        spectral_dict_nogreen_oos[k].to_excel(writer, sheet_name="spectral_oos %s" % k)


# PLOTS
# call function only after both parts (green and nongreen) have been run
plot_SRM(k_list,path=directory_path,folder='/plots/spectral')




