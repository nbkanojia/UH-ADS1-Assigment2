# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:51:59 2023

@author: Nisarg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(filename):
    """
    This function accept the filename and reads csv in worldbank data formate \
    and processes and prepare two dataframes by yeas as columns and country \
    as columns.

    Parameters
    ----------
    filename : string
        input the file name.

    Returns
    -------
    pandas dataframe
        return two dataframes.

    """
    start_from_yeart = 1990
    end_to_year = 2020
   # countrie_list = ["Brazil","Indonesia","Russia","Argentina","Paraguay","Bolivia","Nigeria"]
    countrie_list = ["Brazil","Indonesia","Argentina","Paraguay","Bolivia","Nigeria"]

    # read csv using pandas
    wb_df = pd.read_csv(filename,
                        skiprows=3, iterator=False)

    years_column_list = np.arange(
        start_from_yeart, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    #print("----------------------------",filename)
   #print(wb_df.sort_values("1990",ascending=False)[["Country Name","1990"]].dropna().iloc[:50,:])
    #print("----------------------------")
    # Filter data: select only specific countries and years
    df_selected = wb_df.loc[
                            wb_df["Country Name"].isin(countrie_list),
                            all_cols_list]

    # Transpose
    df_t = pd.DataFrame.transpose(df_selected)
    df_t.columns = df_t.iloc[0]

    # remove first row
    df_t = df_t[1:]
    df_t.index = df_t.index.astype(int)
    df_t.dropna()

    return df_t, df_t.T

def normalise_data(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df_n : TYPE
        DESCRIPTION.

    """
    #dfmin = np.min(np.min(df))
    #dfmax = np.max(np.max(df))
    dfmin = np.min(df)
    dfmax = np.max(df)
    df_n = (df-dfmin)/(dfmax-dfmin) * 100
    return df_n


#def marge_country_with_other_parameter(df, country, )

###### Main Function ################

frst_lnd_data_yw, frst_lnd_data_cw = \
    read_data("API_AG.LND.FRST.ZS_DS2_en_csv_v2_5994693.csv")
    
green_gas_data_yw, green_gas_data_cw = \
    read_data("API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_5995567.csv")
    
gdp_data_yw, gdp_data_cw = \
        read_data("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6011335.csv")
        
co2_data_yw, co2_data_cw = \
        read_data("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5994970.csv")




fig, axes = plt.subplots(1, 2,figsize=(30, 10))

co2_data_yw_n = normalise_data(co2_data_yw)

ap = frst_lnd_data_yw.plot( title="forest_lnd",ax=axes[0])
ap.set_ylim(0,100)
gg = co2_data_yw_n.rolling(5).mean().plot( title="co2",ax=axes[1])
gg.set_ylim(0,100)

plt.show()
