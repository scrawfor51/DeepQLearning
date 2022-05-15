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
    


"""
Helper function to calculate the Relative Strength Index for the date range. 

Adapted from vectorized code handout. 

@param dataframe: A dataframe of the stocks to calculate the RSI for.
@param window_size: The time slice that the RSI should be calculated using. 
@return a dataframe of the RSI for each of the stocks in the dataframe over the time period. 
"""
def Relative_Strength_Index(dataframe, window_size=14):
    
    rsi = dataframe.copy()
    daily_returns = dataframe.copy()
    daily_returns.values[1:,:] = dataframe.values[1:,:] - dataframe.values[:-1, :]
    daily_returns.values[0, :] = np.nan
    
    up_returns = daily_returns[daily_returns >= 0].fillna(0).cumsum()
    down_returns = -1 * daily_returns[daily_returns < 0].fillna(0).cumsum()
    
    up_day_gains = dataframe.copy()
    up_day_gains.loc[:,:] = 0
    up_day_gains.values[window_size:,:] = up_returns.values[window_size:,:] - up_returns.values[:-window_size,:]

    down_day_losses = dataframe.copy()
    down_day_losses.loc[:,:] = 0
    down_day_losses.values[window_size:,:] = down_returns.values[window_size:,:] - down_returns.values[:-window_size,:]
    
    relative_strength = (up_day_gains / window_size) / (down_day_losses / window_size)
    rsi = 100 - (100/ (1 + relative_strength))
    rsi.iloc[:window_size, :] = np.nan
    
    rsi[rsi == np.inf] = 100
    
    return rsi

"""
A helper function used to calculate the Aroon oscillator using a Aroon Indicator. 
Measures strength of trend and likelihood to continue. 

Found on Stackoverflow as posted by @user TheAfricanQuant https://stackoverflow.com/questions/47950466/how-to-build-aroon-indicator-with-python-pandas

@param dataframe: The data for the stocks whose oscillation we want to determine.
@param window_size: The number of periods to look at for the indicators
@return A dataframe containing the Aroon Up and Aroon Down values as well as the Aroon Oscillator value itself. 
"""
def Aroon_Oscillator(dataframe, window_size=25): 
    
   aroon = dataframe.copy()
   aroon.columns = ['Price']
   aroon['Up'] = (100 * aroon['Price'].rolling(window_size + 1).apply(lambda x : x.argmax()) / window_size).values
   aroon['Down'] = (100 * aroon['Price'].rolling(window_size + 1).apply(lambda x : x.argmin()) / window_size).values
    
   return aroon

"""
A helper function used to calculate the stochastic oscillatorr of a given dataframe. 


@param dataframe: The data for the stocks we want to calculate the oscillator for.
@param window_size: The time period to determine the oscillator with regard to. 
@return A dataframe of the oscillator values for the stocks over the range. 
"""
def Stochastic_Oscillator(dataframe, window_size=14):
    
    numerator = (dataframe - dataframe.rolling(window_size, min_periods=window_size).min())
    denominator = (dataframe.rolling(window_size, min_periods=window_size).max()- dataframe.rolling(window_size, min_periods=window_size).min())
    oscillator = 100 * (numerator / denominator)
    
    
    return oscillator 

    
    
    


