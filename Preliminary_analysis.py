import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

path = os.getcwd()
df_stocks=pd.read_excel(path + '\\input_data.xlsx')
#                                         RETURNS
#Create a new dataframe for returns
df_returns = df_stocks
#Compute log-returns: create new columns for each time series of log-returns
df_returns['R_BBGB'] = np.log(df_returns.BBGB/df_returns.BBGB.shift(1))
df_returns['R_SOLGB'] = np.log(df_returns.SOLGB/df_returns.SOLGB.shift(1))
df_returns['R_BBBOND'] = np.log(df_returns.BBBOND/df_returns.BBBOND.shift(1))
df_returns['R_MSCI'] = np.log(df_returns.MSCI/df_returns.MSCI.shift(1))
df_returns['R_SPGSEN'] = np.log(df_returns.SPGSEN/df_returns.SPGSEN.shift(1))
df_returns['R_SP5IAIR'] = np.log(df_returns.SP5IAIR/df_returns.SP5IAIR.shift(1))
df_returns['R_SP5EHCR'] = np.log(df_returns.SP5EHCR/df_returns.SP5EHCR.shift(1))
df_returns['R_SPEUIT'] = np.log(df_returns.SPEUIT/df_returns.SPEUIT.shift(1))

#Remove the first line as it's not a number
df_returns = df_returns.drop([0])

#Define functions for the analysis
def descriptive_statistics_tables(r1, r2, r3, r4, r5, r6, r7, r8):
    statistics = ['Number of Observations','Mean','Minimum','Maximum','Std. Deviation', 'Skewness', 'Kurtosis', 'Jarque-Bera Statistic Test','J-B Test P-Value']
    df = pd.DataFrame(index=statistics)
    df['BBGB'] = 0
    df['SOLGB'] = 0
    df['BBBOND'] = 0
    df['MSCI'] = 0
    df['SPGSEN'] = 0
    df['SP5IAIR'] = 0
    df['SP5EHCR'] = 0
    df['SPEUIT'] = 0
    #r1, r2, r3, r4 = np.array(r1), np.array(r2), np.array(r3), np.array(r4)
    l = [r1, r2, r3, r4, r5, r6, r7, r8]
    i=0
    for asset in l:
        jb_test = scs.jarque_bera(asset)
        stat_arr = np.empty([9,1])
        stat_arr[0] = asset.shape[0]
        stat_arr[1] = asset.mean()
        stat_arr[2] = asset.min()
        stat_arr[3] = asset.max()
        stat_arr[4] = asset.std()
        stat_arr[5] = asset.skew()
        stat_arr[6] = asset.kurtosis()
        stat_arr[7] = jb_test[0]
        stat_arr[8] = jb_test[1]
        df.iloc[:,i] = stat_arr
        i=i+1

    return df
# ---------------------------------------------------------------------------------------------------
#                      ENTIRE PERIOD ANALYSIS
#STATISTICS TABLE
descrstat_all = descriptive_statistics_tables(df_returns.R_BBGB, df_returns.R_SOLGB,df_returns.R_BBBOND,df_returns.R_MSCI, df_returns.R_SPGSEN, df_returns.R_SP5IAIR, df_returns.R_SP5EHCR, df_returns.R_SPEUIT)

#CORRELATION MATRIX
corr_all = df_returns.iloc[:, [9,10,11,12,13]].corr()


#-----------------------------------------------------------------------------------------------------
#                      PRELIMINARY SUBPERIOD ANALYSIS
#Build subperiod dataframes
#BEAR: 05/21/2015 to 04/11/2016 (American date format)
bearmarketDF = df_returns[154:386].copy()
#BULL: 12/01/2016 to 01/31/2018
bullmarketDF = df_returns[553:856].copy()
#PRE-PANDEMIC: 01/03/2019 to 03/03/2020
prepandemicDF = df_returns[1095:1398].copy()
#PANDEMIC 03/04/2020 to 06/07/2021
pandemicDF = df_returns[1398:].copy()

#STATISTICS TABLES
descrstat_bear = descriptive_statistics_tables(bearmarketDF.R_BBGB, bearmarketDF.R_SOLGB ,bearmarketDF.R_BBBOND, bearmarketDF.R_MSCI,
                                               bearmarketDF.R_SPGSEN, bearmarketDF.R_SP5IAIR, bearmarketDF.R_SP5EHCR, bearmarketDF.R_SPEUIT)
descrstat_bull = descriptive_statistics_tables(bullmarketDF.R_BBGB, bullmarketDF.R_SOLGB,bullmarketDF.R_BBBOND,bullmarketDF.R_MSCI,
                                               bullmarketDF.R_SPGSEN, bullmarketDF.R_SP5IAIR, bullmarketDF.R_SP5EHCR, bullmarketDF.R_SPEUIT)
descrstat_prep = descriptive_statistics_tables(prepandemicDF.R_BBGB, prepandemicDF.R_SOLGB, prepandemicDF.R_BBBOND, prepandemicDF.R_MSCI,
                                              prepandemicDF.R_SPGSEN, prepandemicDF.R_SP5IAIR, prepandemicDF.R_SP5EHCR, prepandemicDF.R_SPEUIT)
descrstat_pan = descriptive_statistics_tables(pandemicDF.R_BBGB, pandemicDF.R_SOLGB,pandemicDF.R_BBBOND,pandemicDF.R_MSCI,
                                              pandemicDF.R_SPGSEN, pandemicDF.R_SP5IAIR, pandemicDF.R_SP5EHCR, pandemicDF.R_SPEUIT)
#CORRELATION MATRICES
#BEAR MARKET
corr_bear = bearmarketDF.iloc[:, [9,10,11,12,13]].corr()
#BULL MARKET
corr_bull = bullmarketDF.iloc[:, [9,10,11,12,13]].corr()
#PREPANDEMIC
corr_prep = prepandemicDF.iloc[:, [9,10,11,12,13,14,15,16]].corr()
#PANDEMIC
corr_pan = pandemicDF.iloc[:, [9,10,11,12,13,14,15,16]].corr()

#----------------------------------------------------------------------------------------------------------------
with pd.ExcelWriter("excel_tables/partialtables/preliminary_statistics.xlsx") as writer:
    descrstat_all.to_excel(writer, sheet_name='stat_all')
    descrstat_bear.to_excel(writer, sheet_name='stat_bear')
    descrstat_bull.to_excel(writer, sheet_name='stat_bull')
    descrstat_prep.to_excel(writer, sheet_name='stat_prep')
    descrstat_pan.to_excel(writer, sheet_name='stat_pan')
    corr_all.to_excel(writer, sheet_name='corr_all')
    corr_bear.to_excel(writer, sheet_name='corr_bear')
    corr_bull.to_excel(writer, sheet_name='corr_bull')
    corr_prep.to_excel(writer, sheet_name='corr_prep')
    corr_pan.to_excel(writer, sheet_name='corr_pan')





#----------------------------------------------------------------------------------------------------
# # Stock index plot
# fig = plt.figure(figsize=(10,6))
# plt.plot(df_stocks.Date, df_stocks.MSCI,'y',linewidth=1)
# plt.axvline(x=pd.to_datetime('2015-05-05 00:00:00'), linewidth=0.8)
# plt.axvline(x=pd.to_datetime('2016-04-11 00:00:00'), linewidth=0.8)
# plt.axvline(x=pd.to_datetime('2016-12-01 00:00:00'), linewidth=0.8, color='g')
# plt.axvline(x=pd.to_datetime('2018-01-31 00:00:00'), linewidth=0.8, color='g')
# plt.axvline(x=pd.to_datetime('2020-02-12 00:00:00'), linewidth=0.8, color='darkred')
# plt.axvline(x=pd.to_datetime('2019-01-02 00:00:00'), linewidth=0.8, color='red')
# fig.tight_layout()
# plt.show()
# fig.savefig('plots/preliminary_analysis/MSCI subperiods.png')

# #----------------------------------------------------------------------------------------------------
# # Pairwise plots
# # BBGB and BBBOND
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date, df_stocks.BBGB, 'limegreen', linewidth=0.5, label='BB green bond')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date, df_stocks.BBBOND, 'r', linewidth=0.5, label='Bond')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()
# fig.savefig('plots/preliminary_analysis/BBGB e BOND.png')

# # BBGB and MSCI
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date, df_stocks.BBGB, 'limegreen', linewidth=0.5, label='BB green bond')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date, df_stocks.MSCI, 'gold', linewidth=0.5, label='MSCI')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()
# fig.savefig('plots/preliminary_analysis/BBGB e MSCI.png')

# # BBGB and SPGSEN
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date, df_stocks.BBGB, 'limegreen', linewidth=0.5, label='BB green bond')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date, df_stocks.SPGSEN, linewidth=0.5, label='SPGSEN')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()
# fig.savefig('plots/preliminary_analysis/BBGB e SPGSEN.png')

# # SOLGB and BBBOND
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date, df_stocks.SOLGB, 'forestgreen', linewidth=0.5, label='SOL green bond')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date, df_stocks.BBBOND, 'gold', linewidth=0.5, label='Bond')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()
# fig.savefig('plots/preliminary_analysis/SOLGB e BOND.png')

# # SOLGB and MSCI
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date, df_stocks.SOLGB, 'forestgreen', linewidth=0.5, label='SOL green bond')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date, df_stocks.MSCI, 'r', linewidth=0.5, label='MSCI')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()
# fig.savefig('plots/preliminary_analysis/SOLGB e MSCI.png')

# # SOLGB and SPGSEN
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date, df_stocks.SOLGB, 'forestgreen', linewidth=0.5, label='SOL green bond')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date, df_stocks.SPGSEN, linewidth=0.5, label='SPGSEN')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()
# fig.savefig('plots/preliminary_analysis/SOLGB e SPGSEN.png')

# # BBGB and SP5IAIR
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date[1095:], df_stocks.BBGB[1095:], 'green', linewidth=0.5, label='BBGB')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date[1095:], df_stocks.SP5IAIR[1095:], 'yellow', linewidth=0.5, label='AIR')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()

# # BBGB and SP5EHCR 
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date[1095:], df_stocks.BBGB[1095:],  'green', linewidth=0.5, label='BBGB')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date[1095:], df_stocks.SP5EHCR[1095:], 'blue' ,linewidth=0.5, label='HEALTHCARE')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()

# # BBGB and SPEUIT 
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date[1095:], df_stocks.BBGB[1095:], 'green', linewidth=0.5, label='BBGB')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date[1095:], df_stocks.SPEUIT[1095:], 'red', linewidth=0.5, label='IT')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()

# # SOLGB and SP5IAIR
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date[1095:], df_stocks.SOLGB[1095:],'green',  linewidth=0.5, label='SOL green bond')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date[1095:], df_stocks.SP5IAIR[1095:],'yellow', linewidth=0.5, label='AIR')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()

# # SOLGB and SP5EHCR
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date[1095:], df_stocks.SOLGB[1095:], 'green', linewidth=0.5, label='SOL green bond')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date[1095:], df_stocks.SP5EHCR[1095:],'blue', linewidth=0.5, label='IT')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()

# # SOLGB and SPEUIT
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# lns1 = ax.plot(df_stocks.Date[1095:], df_stocks.SOLGB[1095:], 'green', linewidth=0.5, label='SOL green bond')
# ax2 = ax.twinx()
# lns2 = ax2.plot(df_stocks.Date[1095:], df_stocks.SPEUIT[1095:],'red', linewidth=0.5, label='IT')
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc = 0)
# fig.tight_layout()
# plt.show()

#----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
#                   RETURNS
# #Plots of log-returns
# ret1 = plt.figure(figsize=(8,4))
# plt.plot(df_returns.Date, df_returns.R_BBGB, 'forestgreen', linewidth=0.6)
# plt.grid(linewidth=0.3)
# plt.tight_layout()
# plt.show()
# ret1.savefig('plots/preliminary_analysis/RET BBGB.png')

# ret2 = plt.figure(figsize=(8,4))
# plt.plot(df_returns.Date, df_returns.R_BBBOND, 'r', linewidth=0.8)
# plt.grid(linewidth=0.3)
# plt.tight_layout()
# plt.show()
# ret2.savefig('plots/preliminary_analysis/RET BOND.png')

# ret3 = plt.figure(figsize=(8,4))
# plt.plot(df_returns.Date, df_returns.R_MSCI, 'y', linewidth=0.8)
# plt.grid(linewidth=0.3)
# plt.tight_layout()
# plt.show()
# ret3.savefig('plots/preliminary_analysis/RET MSCI.png')

# ret4 = plt.figure(figsize=(8,4))
# plt.plot(df_returns.Date, df_returns.R_SPGSEN, linewidth=0.8)
# plt.grid(linewidth=0.3)
# plt.tight_layout()
# plt.show()
# ret4.savefig('plots/preliminary_analysis/RET SPGSEN.png')

# ret5 = plt.figure(figsize=(8,4))
# plt.plot(df_returns.Date, df_returns.R_SOLGB, 'limegreen', linewidth=0.8)
# plt.grid(linewidth=0.3)
# plt.tight_layout()
# plt.show()
# ret3.savefig('plots/preliminary_analysis/RET SOLGB.png')

#----------------------------------------------------------------------------------------------------
