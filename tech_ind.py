#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:46:33 2022

@author: Stephen Crawford 
"""

import os
import pandas as pd 
from matplotlib import pyplot as plt 
import math
import numpy as np

# use ./data
# in sample of 1/1/2018-12/31/2019
# out of sample of 1/1/2020-12/31/2021
# start with 200k -- can go negative
# allowable orders are +/- 2000, 1000
# only trade Disney (DIS) 
# Pos limited to +1000, 0, -1000 shares
# baseline of bought and held 1000 shares for period 
# consider on li sample when devising and fine tuning; only eval once after strategy is finalized 
# make a tweaked version of bascktest.py taht takes a df_trades DataFrame instaed of CSV

# each indicator takes in historical price &/| volume data and computes the technical indicator 
# need at least three indicators--one myst not be Moving Average, Bollinger Bands, or RSI

        

"""
Helper function to read data files of requested stocks and extract date range 

@param start_date: The start of the date range you want to extract
@parem end_date: The end of the date range you want to extract
@param symbols: A list of stock ticker symbols
@param column_name: A single data column to retain for each symbol 
@param include_spy: Controls whether prices for SPY will be retained in the output 
@return A dataframe of the desired stocks and their data
"""
def get_data(start_date, end_date, symbols, column_name = 'Adj Close', include_spy=True):
    

    standardized_symbols = []
    #Catch lowercase symbols
    
    if include_spy:
        standardized_symbols.append('SPY')
    for symbol in symbols:
        standardized_symbols.append(symbol.upper())
        
        
    queried_data = pd.DataFrame(columns=standardized_symbols) #make an empty dataframe as a series of ticker name
    data_path = './data'
    
    for file in os.listdir(data_path):
         if file[:file.find('.')] in standardized_symbols: 
             df = pd.read_csv(os.path.join(data_path, file), index_col='Date', parse_dates=True, float_precision=(None))
             df = df.loc[start_date : end_date]
             queried_data[file[:file.find('.')]] = df[column_name]
             
    return(queried_data)



"""
Helper function to calculate the Simple Moving Average of a dataframe.

Adapted from the vectorized code handout. 

@param dataframe: A dataframe of the price history of the stocks whose SMA we wish to calculate.
@param window_size: The number of days to include in the SMA widnow. 
@return a dataframe with the SMA-window_size for each stock in the original dataframe. 
"""
def SMA(dataframe, window_size):
    
    sma_df = dataframe.rolling(window=window_size, min_periods=window_size).mean()
    
    return sma_df

"""
Helper function to get the SMA ratio.

Adapted from the vectorized code handout. 

@param dataframe: A dataframe of the price history of the stocks whose SMA we wish to calculate.
@param window_size: The number of days to include in the SMA widnow. 
@return a dataframe consisting of the price divided by the SMA. 
"""
def SMA_ratio(dataframe, window_size):
    
    sma_df = SMA(dataframe, window_size)
    ratio = dataframe/sma_df
    
    return ratio
    
    
"""
Helper function to calculate the Bollinger Bands of a dataframe. 

Adapted from vectorized code handout.

@param dataframe: The price history of the stocks whose Bollinger Bands we want to calculate 
@param band_range: The scaling value for the range of the bands--how many standard deviations they should be from SMA
@return dataframe with the top and bottom Bollinger Bands as additional columns 
"""
def Bollinger_Bands(dataframe, window_size, band_range=2):
    
    sma_df = dataframe.copy()
    sma_df.columns = ["Price"]
    sma_df["SMA"] = sma_df["Price"].rolling(window=window_size, min_periods=window_size).mean()
    rolling_std = sma_df["Price"].rolling(window=window_size, min_periods=window_size).std()
    top_band = sma_df["SMA"] + (band_range * rolling_std)
    bottom_band = sma_df["SMA"] - (band_range * rolling_std)
    sma_df["Top Band"] = top_band
    bb = sma_df
    bb["Bottom Band"] = bottom_band
   
    return bb


"""
Helper function to calculate the Bollinger Bands Percentage of a dataframe. 

Adapted from vectorized code handout.

@param dataframe: The price history of the stocks whose Bollinger Bands we want to calculate 
@param band_range: The scaling value for the range of the bands--how many standard deviations they should be from SMA
@return dataframe of the Bollinger Band percentage
"""
def Bollinger_Bands_Percentage(dataframe, window_size, band_range=2):
    
    bb_sma_df = Bollinger_Bands(dataframe, window_size, band_range)
    

    bb_sma_df["BB Percentage"] = (bb_sma_df["Price"] - bb_sma_df["Bottom Band"]) / (bb_sma_df["Top Band"] - bb_sma_df["Bottom Band"])
    return bb_sma_df


"""
A helper function used to calculate the on-balance volume indicator. 
Measures the pos. & neg. flow of volume over time.


REQUIRES VOLUME DATA AS WELL

@param dataframe: The price AND volume data for the stocks whose on-balance volume we wish to calculate. 
@return A dataframe of on-balance volume history for the stocks 
"""
def On_Balance_Volume(price_dataframe, volume_dataframe):
    
    #If OBV up--buyers are willing to push price higher 
    # Down means selling outpacing buying 
    
    obv = price_dataframe.copy()
    obv.values[0] = 0
    test_df = volume_dataframe.copy()
    test_df['Price'] = price_dataframe.values
   
    
    daily_returns = price_dataframe.copy()
    daily_returns.values[1:,:] = price_dataframe.values[1:,:] - price_dataframe.values[:-1, :]
    daily_returns.values[0, :] = 0
    test_df['Daily_Returns'] = daily_returns.values
   
    
    daily_returns.values[daily_returns.values < 0] = -1
    
    daily_returns.values[daily_returns.values > 0] = 1
    daily_returns.values[daily_returns.values == 0] = 0

    
    obv = (daily_returns * volume_dataframe)
    obv = obv.cumsum()
    
    return obv
    
    
"""
A helper function used to calculate the William's Precentage Range for a dataframe.

@param dataframe: The data for the stocks we want to calculate for.
@param window_size: The amount of periods to consider at a time
@return A dataframe of the Willaims Percentage Range (0 to -100) for the given dataframe.
"""
def Williams_Percentage_Range(dataframe, window_size=14):
    
    price = dataframe.copy()
    numerator = (price.rolling(window_size, min_periods=window_size).max() - price)
    denominator = (price.rolling(window_size, min_periods=window_size).max() - price.rolling(window_size, min_periods=window_size).min())
    price['Williams Percentage'] = (numerator/denominator) * -100
    return price
    
    
    
    


