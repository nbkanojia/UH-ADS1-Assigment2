# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:51:59 2023

@author: Nisarg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_world_bank_csv(filename):
    """
    Accept the filename and reads csv in worldbank data formate. 

    and processes and prepare two dataframes by yeas as columns and country 
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
    countrie_list = ["Brazil", "Indonesia", "Russian Federation", "Argentina",
                     "Paraguay", "Bolivia", "Nigeria"]

    # read csv using pandas
    wb_df = pd.read_csv(filename,
                        skiprows=3, iterator=False)
    wb_df.dropna(axis=1)
    years_column_list = np.arange(
        start_from_yeart, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # Filter data: select only specific countries and years
    df_country_index = wb_df.loc[
        wb_df["Country Name"].isin(countrie_list),
        all_cols_list]
    
    #make country as index
    df_country_index.index = df_country_index["Country Name"]
    # remove first row as it is become index
    df_country_index.drop("Country Name", axis=1, inplace=True)
   
    #convert year columns as interger
    df_country_index.columns = df_country_index.columns.astype(int)

    # Transpose
    df_year_index = pd.DataFrame.transpose(df_country_index)
   
    return df_year_index, df_country_index


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
    # dfmin = np.min(np.min(df))
    # dfmax = np.max(np.max(df))
    dfmin = df.min()
    dfmax = df.max()
    df_n = (df-dfmin)/(dfmax-dfmin) * 100
    return df_n


def print_statics(title, df, country="All"):
    """
    

    Parameters
    ----------
    title : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if(country!="All"):
        df = df[country]
        title = title+"(" + country +")"
    print("\n====",  (title+" ").ljust(65, "=")  )
    
    
    
    # print describe function,it will display count, mean, std, min..etc 
    df_describe = df.describe()
    
    # add two other statics method kurtosis 
    # to see the tailedness vs heavy tails 
    # and skew to see skewness(Negative, Positive , Zero)
    df_describe.loc["kurtosis"]  = df.kurtosis()
    df_describe.loc["skew"]  = df.skew()
    df_describe.loc["median"]  = df.median()

    print(df_describe)
    print("".ljust(70, "=") ,"\n")


def plot_and_save_line_chart(fig_name, title, xlabel, ylabel, df):
    """
    

    Parameters
    ----------
    fig_name : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(10, 6))
    ap = df.plot(title=title)
    # ap.set_ylim(0,100)
    
    ap.set_xlabel(xlabel)
    ap.set_ylabel(ylabel)
    fig = ap.get_figure()
    fig.savefig(fig_name, dpi=300)


def plot_and_save_bar_chart(fig_name, title, xlabel, ylabel, df):
    """
    

    Parameters
    ----------
    fig_name : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(10, 6))
    ap = df.plot.bar(title=title)
    # ap.set_ylim(0,100)
    ap.set_xlabel(xlabel)
    ap.set_ylabel(ylabel)
    fig = ap.get_figure()
    fig.savefig(fig_name, dpi=300)
    
    
def plot_and_save_heat_map(country, dict_label_dataset):
    """


    Parameters
    ----------
    title : TYPE
        DESCRIPTION.
    country : TYPE
        DESCRIPTION.
    dict_label_dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(10, 6))
    result = pd.DataFrame()
    for lbl in dict_label_dataset:
        result[lbl] = dict_label_dataset[lbl][[
            country]].values.flatten().tolist()

    corr = result.corr(numeric_only=True).round(2)
    plt.figure()
    plt.imshow(corr, vmin=-1, vmax=1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(1, -1.25, -0.25))  # Set specific tick positions

    for i in range(0, len(dict_label_dataset)-1):
        plt.axhline(i + 0.5, color='black', linewidth=1)
        plt.axvline(i + 0.5, color='black', linewidth=1)

    for (i, j), z in np.ndenumerate(corr):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    plt.title(country)
    plt.savefig(country+".png", dpi=300)


###### Main Function ################

# read csv files and get the dataframs
frst_lnd_data_yw, frst_lnd_data_cw = \
    read_world_bank_csv("API_AG.LND.FRST.ZS_DS2_en_csv_v2_5994693.csv")

gdp_data_yw, gdp_data_cw = \
    read_world_bank_csv("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6011335.csv")

co2_data_yw, co2_data_cw = \
    read_world_bank_csv("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5994970.csv")

total_population_data_yw, total_population_data_cw = \
    read_world_bank_csv("API_SP.POP.TOTL_DS2_en_csv_v2_6011311.csv")

ele_data_yw, ele_population_data_cw = \
    read_world_bank_csv("API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_5995551.csv")

agri_lnd_yw, agri_lnd_cw = \
    read_world_bank_csv("API_AG.LND.AGRI.ZS_DS2_en_csv_v2_5995314.csv")

arab_lnd_yw, arab_lnd_cw = \
    read_world_bank_csv("API_AG.LND.ARBL.ZS_DS2_en_csv_v2_5995308.csv")

# print statics
print_statics("Forest land statics",frst_lnd_data_yw,"Argentina")
print_statics("CO2",co2_data_yw)

# plot chats
plot_and_save_line_chart("forest.png", "Forest area (% of land area)", "Year",
                         "%", frst_lnd_data_yw)

co2_data_yw_n = normalise_data(co2_data_yw)
plot_and_save_line_chart("co2.png", "CO2 emissions (kt)", "Year", "kt",
                         co2_data_yw_n)


plot_and_save_line_chart("electric.png",
                         "Electric power consumption (kWh per capita)", "Year",
                         "kWh", ele_data_yw)
plot_and_save_line_chart("population.png", "Population, total", "Year", "",
                         ele_data_yw)

# prepare heat map dictonary label with it's dataframe
dict_heat_map = {
    "forest land": frst_lnd_data_yw,
    "co2": co2_data_yw,
    "gdp": gdp_data_yw,
    "total population": total_population_data_yw,
    "electricity": ele_data_yw,
    "Agricultural land": agri_lnd_yw,
    "Arable land(%)": arab_lnd_yw
}

#plot heat map
plot_and_save_heat_map("Argentina", dict_heat_map)
plot_and_save_heat_map("Brazil", dict_heat_map)
plot_and_save_heat_map("Nigeria", dict_heat_map)

print(gdp_data_yw)


plt.show()

