The following code was written on Spyder, with Python 3.8, in the framework of Anaconda 3.
The Python libraries and modules required are numpy, pandas, os, matplotlib, statsmodels, win32com, math, scipy, datetime and empyrical.

The data is recovered from the Bloomberg terminal and refers to:
- Bloomberg Barclays MSCI Green Bond Index (BBGB)
- Solactive Green Bond Index (SOLGB)
- MSCI World Index (MSCI)
- Bloomberg Barclays Global Aggregate Total Return Index (BBBOND)
- S&P GSCI Energy Index  (SPGSEN)
- S&P 500 Airlines (SP5IAIR)
- S&P 500 Health Care (SP5EHCR)
- S&P 500 Euro Information Technology (SPEUIT) 

The code relies on the main folder (EnergyEconomics) to be set as the directory.
The results of the entire paper can be obtained by running the script "ENTIRE_PAPER.py" (pressing F5 once it's open, on most computers). This file calls three other scripts, representing each of the chapters corresponding to their names:
- Preliminary_analysis.py
- Market_relationships_analysis.py
- Asset_allocation_analysis.py
They include commented and detailed explanations of each step. Other scripts that are used in computation are present in the folder "modules".
Plots are automatically saved in the folder "plots" and are divided in sub-folders corresponding to their name.
Data is automatically exported to the folder "\excel_tables\partialtables". The tables present in the paper are obtained in the excel files of the following names:
- preliminary_analysis.xlsx
- market_relationships_analysis.xlsx
- asset_allocation_analysis.xlsx
Excel file links might need to be updated by setting them to the new file path, through Data>Edit links.
The tables are found in the Excel file which is named after the paragraph that they appear in.

A guide for where to find each table:
Preliminary Analysis
Table 1: preliminary_analysis.xlsx, sheet: descriptive_statistics
Table 2: preliminary_analysis.xlsx, sheet: descriptive_statistics
Table 3: preliminary_analysis.xlsx, sheet: descriptive_statistics

Analysis of Market Relationships
Table 4: market_relationships_analysis.xlsx, sheet: DCC_stats
Table 5: market_relationships_analysis.xlsx, sheet: DCC_stats

Asset Allocation Analysis
Table 6: asset_allocation_results.xlsx, sheet: Weights
Table 7: asset_allocation_results.xlsx, sheet: Weights
Table 8: asset_allocation_results.xlsx, sheet: Bear period
Table 9: asset_allocation_results.xlsx, sheet: Bull period
Table 10: asset_allocation_results.xlsx, sheet: Prepandemic
Table 11: asset_allocation_results.xlsx, sheet: Pandemic

Appendix A
Table 12: market_relationships_analysis.xlsx, sheet: marginal_parameters
Table 13: market_relationships_analysis.xlsx, sheet: marginal_parameters
Table 14: market_relationships_analysis.xlsx, sheet: marginal_parameters
Table 15: market_relationships_analysis.xlsx, sheet: marginal_parameters

Appendix B
Table 16: market_relationships_analysis.xlsx, sheet: all 95%_UNC_CORR_df4
Table 17: market_relationships_analysis.xlsx, sheet: all 95%_UNC_CORR_correctDFs

Appendix C
Table 18: market_relationships_analysis.xlsx, sheet: GARCH
Table 19: market_relationships_analysis.xlsx, sheet: GARCH
Table 20: market_relationships_analysis.xlsx, sheet: GARCH
Table 21: market_relationships_analysis.xlsx, sheet: GARCH
Table 22: market_relationships_analysis.xlsx, sheet: GARCH
Table 23: market_relationships_analysis.xlsx, sheet: DCC
Table 24: market_relationships_analysis.xlsx, sheet: DCC

Appendix E
Table 25: asset_allocation_results.xlsx, sheet: Spectral_green
Table 26: asset_allocation_results.xlsx, sheet: Spectral_nongreen
Table 27: asset_allocation_results.xlsx, sheet: Spectral_green
Table 28: asset_allocation_results.xlsx, sheet: Spectral_nongreen




end

