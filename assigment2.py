# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:51:59 2023

@author: Nisarg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp


def read_world_bank_csv(filename):
    """
    Accept the csv filename with worldbank data format.
    Read a file and processes and prepare two dataframes by yeas as index
    and country as an index.

    Parameters
    ----------
    filename : string
        input the csv file name..

    Returns
    -------
    df_year_index : pandas.DataFrame
        DataFrame with years as an index.
    df_country_index : pandas.DataFrame
        DataFrame with the country as an index.

    """
    # set year range and country list to filter the dataset
    start_from_yeart = 1990
    end_to_year = 2020
    countrie_list = ["Brazil", "Indonesia", "Russian Federation", "Argentina",
                     "Paraguay", "Bolivia", "Nigeria"]

    # read csv using pandas
    wb_df = pd.read_csv(filename,
                        skiprows=3, iterator=False)

    # clean na data, remove columns
    wb_df.dropna(axis=1)

    # prepare a column list to select from the dataset
    years_column_list = np.arange(
        start_from_yeart, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # filter data: select only specific countries and years
    df_country_index = wb_df.loc[
        wb_df["Country Name"].isin(countrie_list),
        all_cols_list]

    # make the country as index and then drop column as it becomes index
    df_country_index.index = df_country_index["Country Name"]
    df_country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns as interger
    df_country_index.columns = df_country_index.columns.astype(int)

    # Transpose dataframe and make the country as an index
    df_year_index = pd.DataFrame.transpose(df_country_index)

    # return the two dataframes year as index and country as index
    return df_year_index, df_country_index


def normalise_data(df_to_normal):
    """
    Normalise the dataset within the 0 to 100 range.

    Parameters
    ----------
    df_to_normal : pandas.DataFrame
        dataframe to be normalise

    Returns
    -------
    df_nornal : pandas.DataFrame
        normalise data.

    """
    # get the min and max from the data
    dfmin = df_to_normal.min()
    dfmax = df_to_normal.max()

    # apply formula and normalise data
    df_nornal = (df_to_normal-dfmin)/(dfmax-dfmin) * 100
    return df_nornal


def print_statics_summary(title, df_for_stat, country="All"):
    """
    Shoe statics summary of dataset like describe function output, kurtosis,
    skew and median.

    Parameters
    ----------
    title : string
        Title of the summary.
    df_for_stat : pandas.DataFrame
        dataframe for which statics summary display.
    country : string, optional
        filter summery for specify country. The default is "All".

    Returns
    -------
    None.

    """
    # check if country is passed in parameter, then filter the supplied
    # dataframe else process all country
    if country != "All":
        df_for_stat = df_for_stat[country]
        title = title+"(" + country + ")"
    print("\n====",  (title+" ").ljust(65, "="))

    # print describe function,it will display count, mean, std, min..etc
    df_describe = df_for_stat.describe()

    # add three other statics method kurtosis
    # to see the tailedness vs heavy tails
    # and skew to see skewness(Negative, Positive , Zero) and median
    df_describe.loc["kurtosis"] = sp.kurtosis(df_for_stat)
    df_describe.loc["skew"] = sp.skew(df_for_stat)
    df_describe.loc["median"] = df_for_stat.median()

    # print summary
    print(df_describe)
    print("".ljust(70, "="), "\n")


def plot_and_save_line_chart(df_chart, title, xlabel, ylabel, fig_name):
    """
    Create line chart using df_chart dataframe and save the figure on disk.

    Parameters
    ----------
    df_chart : pandas.DataFrame
        create line chart for df_chart.
    title : string
        title of the chart.
    xlabel : string
        set x label.
    ylabel : string
        set y label.
    fig_name : string
        chart image name.

    Returns
    -------
    None.

    """
    # plot line chart and set xlabel and ylabel
    plt.figure()
    plot = df_chart.plot(title=title)
    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)

    # get figure object from pandas plot and save the image on disk
    fig = plot.get_figure()
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")


def plot_and_save_bar_chart(df_chart, title, xlabel, ylabel, fig_name):
    """
    Create bar chart using df_chart dataframe and save the figure on disk.

    Parameters
    ----------
    df_chart : pandas.DataFrame
        create bar chart for df_chart.
    title : string
        title of the chart.
    xlabel : string
        set x label.
    ylabel : string
        set y label.
    fig_name : string
        chart image name.

    Returns
    -------
    None.

    """
    # select specific years only and plot the bar chart
    plot = df_chart.loc[:, [1990, 2000, 2010, 2020]].plot.bar(title=title,
                                                              rot=45)
    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)

    # get figure object from pandas plot and save the image on disk
    fig = plot.get_figure()
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")


def plot_and_save_stack_bar_chart(df_chart, title, xlabel, ylabel, fig_name):
    """
    Create bar chart using df_chart dataframe and save the figure on disk.

    Parameters
    ----------
    df_chart : pandas.DataFrame
        create bar chart for df_chart.
    title : string
        title of the chart.
    xlabel : string
        set x label.
    ylabel : string
        set y label.
    fig_name : string
        chart image name.

    Returns
    -------
    None.

    """
    # select specific years only and plot the bar chart
    plot = df_chart.loc[:, [1990, 2000, 2010, 2020]]\
        .plot.bar(title=title, stacked=True, rot=45,)
    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)

    # get figure object from pandas plot and save the image on disk
    fig = plot.get_figure()
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")


def plot_and_save_heatmap(country, dict_label_dataset):
    """
    Create heatmap for country with differnt paramters
    and display it's correlation.

    Parameters
    ----------
    country : string
        filter dataset with country.
    dict_label_dataset : dict
        dictionary list of the key pair, where key is parameter name
        and values is dataset.

    Returns
    -------
    None.

    """
    # create blank dataframe
    result = pd.DataFrame()

    # loop through the all paramters in dictionary and prepare dataframe
    for lbl in dict_label_dataset:
        result[lbl] = dict_label_dataset[lbl][[
            country]].values.flatten().tolist()

    # find the correlation between paramters
    corr = result.corr().round(2)

    # create heatmap
    plt.figure()
    plt.imshow(corr, vmin=-1, vmax=1, cmap="Wistia")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    cbar = plt.colorbar()
    # set specific tick positions
    cbar.set_ticks(np.arange(1, -1.25, -0.25))

    # print boxes
    for i in range(0, len(dict_label_dataset)-1):
        plt.axhline(i + 0.5, color='black', linewidth=1)
        plt.axvline(i + 0.5, color='black', linewidth=1)

    # display correlation values in boxes
    for (i, j), corr_val in np.ndenumerate(corr):
        plt.text(j, i, f'{corr_val:0.2f}', ha='center', va='center',
                 color="Black")

    plt.title(country)

    # save file in disk
    plt.savefig(country+".png", dpi=300, bbox_inches="tight")


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

ele_data_yw, ele_data_cw = \
    read_world_bank_csv("API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_5995551.csv")

agri_lnd_yw, agri_lnd_cw = \
    read_world_bank_csv("API_AG.LND.AGRI.ZS_DS2_en_csv_v2_5995314.csv")

arab_lnd_yw, arab_lnd_cw = \
    read_world_bank_csv("API_AG.LND.ARBL.ZS_DS2_en_csv_v2_5995308.csv")

# print statics summary

print_statics_summary("Forest land statics", frst_lnd_data_yw, "Argentina")
print_statics_summary("CO2 emissions", co2_data_yw)
print_statics_summary("Forest area", frst_lnd_data_yw)
print_statics_summary("GDP (current US$)", gdp_data_yw)

# plot charts bar chart for Forest area and CO2emissions
plot_and_save_bar_chart(frst_lnd_data_cw, "Forest area (% of land area)",
                        "", "%", "forest.png")
plot_and_save_bar_chart(co2_data_cw, "CO2 emissions (kt)", "", "kt", "co2.png")


# plot line charts for Electric power consumption and Population
plot_and_save_line_chart(ele_data_yw.loc[1995:2015, :],
                         "Electric power consumption (kWh per capital)",
                         "Year", "kWh", "electric.png")
plot_and_save_line_chart(total_population_data_yw, "Population, total", "Year",
                         "", "population.png")
plot_and_save_line_chart(gdp_data_yw, "GDP (current US$)", "Year", "Trillion",
                         "gdp.png")

# prepare heat map dictonary label with it's dataframe object
dict_heat_map = {
    "forest land": frst_lnd_data_yw,
    "co2": co2_data_yw,
    "gdp": gdp_data_yw,
    "total population": total_population_data_yw,
    "electricity": ele_data_yw,
    "Agricultural land": agri_lnd_yw,
    "Arable land(%)": arab_lnd_yw
}

# plot heat map for differnt countries
plot_and_save_heatmap("Brazil", dict_heat_map)
plot_and_save_heatmap("Indonesia", dict_heat_map)
plot_and_save_heatmap("Russian Federation", dict_heat_map)
plot_and_save_heatmap("Argentina", dict_heat_map)
plot_and_save_heatmap("Paraguay", dict_heat_map)
plot_and_save_heatmap("Bolivia", dict_heat_map)
plot_and_save_heatmap("Nigeria", dict_heat_map)

# show all plots
plt.show()
