# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:51:59 2023

@author: Nisarg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as ptl


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
    countrie_list = ["China", "United States", "India", "Russian Federation",
                 "Germany", "Brazil"]
    
    # read csv using pandas
    wb_df = pd.read_csv(filename,
                         skiprows=3, iterator=False)

    years_column_list = np.arange(
        start_from_yeart, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # Filter data: select only specific countries and years
    df_selected = wb_df.loc[wb_df["Country Name"].isin(countrie_list),
                             all_cols_list]

    # Transpose
    df_t = pd.DataFrame.transpose(df_selected)
    df_t.columns = df_t.iloc[0]

    # remove first row
    df_t = df_t[1:]
    df_t.index = df_t.index.astype(int)


    return df_t , df_t.T


###### Main Function ################

ag_lnd_data,ag_lnd_data_t = read_data("API_AG.LND.FRST.ZS_DS2_en_csv_v2_5994693.csv")
green_gas_data, green_gas_data_t= read_data("API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_5995567.csv")
print(ag_lnd_data,green_gas_data)
print(ag_lnd_data_t,green_gas_data_t)


    