#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:36:30 2022
@author: Stephen & Tugi 
"""

import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  
from TabularQLearner import TabularQLearner
import tech_ind 
import sys
import timeit
import datetime 
from matplotlib import cm
from DeepQLearner import DeepQLearner
 
TRIPS_WITHOUT_DYNA = 500
TRIPS_WITH_DYNA = 50
FAILURE_RATE = 0

class StockEnvironment:

  def __init__ (self, fixed = None, floating = None, starting_cash = None, share_limit = None):
    
    self.shares = share_limit
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = starting_cash
    self.learner = None

  """
  Helper method to calculate all the indicator values for the date range.
  @param Start_date: Start date for indicators
  @param End_date: The last day for the data  
  @param symbol: The stock symbol to calculat the indicators for
  @return: A dataframe of the stock data with all the indicators calculated  
  """
  def prepare_world (self, start_date, end_date, symbol):
  
    price_data = tech_ind.get_data(start_date, end_date, symbols = [symbol], include_spy=False)
   
    volume_data = tech_ind.get_data(start_date, end_date, symbols = [symbol], column_name = 'Volume', include_spy=False)
   
    williams = tech_ind.Williams_Percentage_Range(price_data)
    
    bb = tech_ind.Bollinger_Bands(price_data, 14)
  
    obv = tech_ind.On_Balance_Volume(price_data, volume_data)
    world = price_data.copy()
    world.columns = ['Price']
    world['Williams Percent Range'] = williams['Williams Percentage']
    
    world['Bollinger Band Percentage'] = (world['Price'].sub(bb['SMA'], fill_value=np.nan))/(bb['Top Band'].sub(bb['Bottom Band'], fill_value=np.nan))
    world['OBV Normalized'] = obv/(abs(obv).rolling(window=5, min_periods=5).sum()) * 100 # scaled between -100 and 100
    world = world.ffill()
    world = world.bfill()
    return world
    

  """ 
  Helper function to calculate the state number of a state given specific holdings and the day it is.
  
  @param df: The environmental dataframe with all the stock indicator values
  @param day: The steps in the date range we are on
  @param holdings: How many shares of the stock the trader has.
  @return a number representing the state.
  """
  def calc_state (self, df, day, holdings):
    """ Quantizes the state to a single number. """

   
    current_williams = df.iloc[day, df.columns.get_loc('Williams Percent Range')] # 0 to -100 # 3 buckets 
    current_bbp = df.iloc[day, df.columns.get_loc('Bollinger Band Percentage')] # x < 0 < x < sma < x < top band < x <--buckets 0, 1, 2, 3
    current_obv = df.iloc[day, df.columns.get_loc('OBV Normalized')] 
    state = 0
    
        
    if holdings > 0:  # 3 buckets long, flat, short
        state += 0
    if holdings == 0:
        state += 1
    if holdings < 0:
        state += 2
    
    ws = 0 # 3    buckets oversold, neither, overbought
    if current_williams < -80:
        ws = 0
    elif -80 < current_williams < -20:
        ws = 1
    else:
        ws = 2
        
    state += 3*ws    # add 0, 3, 6     
    
    bbps = 0 # 4 buckets for bbp -- under bottom, between bottom and middle, between middle and top, over top
    if current_bbp < 0:
        bbps = 0
    elif 0 < current_bbp <= 50:
        bbps = 1
    elif 50 < current_bbp < 100:
        bbps = 2
    elif current_bbp > 100:
        bbps = 3
    
    state += 9*bbps # add 0, 9, 18, 27
    
    os = 0 # 4  buckets between -100 and -50, between -50 and 0, between 0 and 50 and greater than 50
    if current_obv <= -50:
        os = 0
    elif -50 < current_obv <= 0:
        os = 1
    elif 0 < current_obv <= 50:
        os = 2
    else:
        os = 3
    
    state += 36*os # 
    
    return int(state)


  """
  A helper function which applies the last day's action and returns the reward and new state.
  @param world: The environment dataframe
  @param Day: The step of training the learner as on.
  @param s: The current state representation  
  @param a: The selected action
  @return the new state s_prime and the reward for the last action r
  """
  def query_world(self, world, day, s, a):
    
    world = world.copy()
    if np.random.random() < FAILURE_RATE: # what if we cannot make a perfect trade everytime? 
      a = np.random.randint(3) 
      
    holdings = 0
    if a == 0:
        holdings = 1000
    if a == 1:
        holdings = 0
    if a == 2:
        holdings = -1000
        
    day_next = day + 1
    
    if day == world.shape[0] - 1:
        day_next = day
        
    s_prime = self.calc_state(world, day_next, holdings)
    
    r = (world.iloc[day, world.columns.get_loc('Portfolio')]/world.iloc[day-1, world.columns.get_loc('Portfolio')] ) - 1
    
    return s_prime, r
  

  """
  A class method for training a leaner.
  
  @param start: The start date for the training
  @param end: End date for training
  @param symbol: The stock symbol to train on 
  @param Trips: The number of trips to train for
  @param Dyna: The dyna halluncination value
  @param eps: The value of epislon for random actions
  @param eps_decay: The rate at which epsilon shoul decay when a random action is taken.
  @return a trained learner 
  """
  def train_learner( self, start = None, end = None, symbol = None, trips = 0, dyna = 0,
                     eps = 0.0, eps_decay = 0.0):
    
    world = self.prepare_world(start, end, symbol)
    
    world_size = 144 # 3 * 3 * 4 * 4
    
    if dyna > 0:
      learner = DeepQLearner(states=world_size, actions = 3, epsilon=eps,epsilon_decay=eps_decay, dyna=dyna)
    else:
      learner = DeepQLearner(states=world_size, actions = 3, epsilon=eps,epsilon_decay=eps_decay)
    
    start = self.calc_state(world, 0, 0)  # start with the new world at the start date and with no positions 
    
    # Remember the total rewards of each trip individually.
    trip_rewards = []
    
    # Each loop is one trip through the state space 
    for i in range(trips):

      # A new trip starts with the learner at the start state with no rewards.
      # Get the initial action.
      world['Cash'] = self.starting_cash
      world['Portfolio'] = world['Cash']
      world['Positions'] = 0
      
      s = start
      trip_reward = 0
      a = learner.test(self.calc_state(world, 0, 0)) # action is long, flat, short -- 0, 1, 2
      steps_remaining = world.shape[0] # Can only move forward in states up until end date
      day_count = 0
    
      # Each loop is one day
      while steps_remaining > 0:
        
        day = world.index[day_count]
      
        holdings = 0
        if a == 0:
            holdings = 1000
        if a == 1:
            holdings = 0
        if a == 2:
            holdings = -1000
      
        
        world.loc[day:, 'Positions'] = holdings
        holdings_change = world.iloc[day_count,  world.columns.get_loc('Positions')] - world.iloc[day_count-1,  world.columns.get_loc('Positions')] # Calculate change in position
    
       
        yesterday_cash = world.iloc[day_count-1, world.columns.get_loc("Cash")]
        today_price = world.iloc[day_count, world.columns.get_loc('Price')]
      
        if holdings_change:
            
            world.loc[day:,'Cash'] = world.loc[day,'Cash'] - (holdings_change * today_price) - abs(holdings_change * today_price)*self.floating_cost - self.fixed_cost 
            world.loc[day:,'Portfolio'] =  world.loc[day:, 'Positions'] * world.loc[day:, 'Price'] + world.loc[day:, 'Cash'] #Update today's portfolio
        
        # Apply the most recent action and determine its reward.
        s, r = self.query_world(world, day_count, s, a) # get a new state and a reward for our action 
      
        
        # Allow the learner to experience what happened.
        a = learner.train(self.calc_state(world, day_count, a), r)
       
        # Accumulate the total rewards for this trip.
        trip_reward += world.iloc[day_count, world.columns.get_loc('Portfolio')] - world.iloc[day_count-1, world.columns.get_loc('Portfolio')] # The numeric value between yesterday's portfolio and today
        
        # Elapse time.
        steps_remaining -= 1
        day_count += 1
      print("For trip ", i, " reward  is: ", trip_reward)
      trip_rewards.append(trip_reward)
        #Breakout when there is convergance (5 days in a row with same trip rewards)
      # if (i > 5 and trip_rewards[-1] == trip_rewards[-2] and trip_rewards[-2] == trip_rewards[-3] and trip_rewards[-3] == trip_rewards[-4] and trip_rewards[-4] == trip_rewards[-5]):
      #     break
     
    for i in range(len(trip_rewards)):
        print("For trip number ", i, " net result is: ", trip_rewards[i])
        
    self.learner = learner
    return learner
    

  """
  A helper method for testing a learner that has already been trained.
  
  @param Start: Start Date
  @param End: End date
  @param Symbol: Stock symbol to trade
  """
  def test_learner( self, start = None, end = None, symbol = None):
    
    world = self.prepare_world(start, end, symbol)
    baseline = world['Price'].copy()
    baseline.iloc[:] = np.nan
    baseline.columns = ['Positions']
    baseline['Positions'] = 1000
    baseline['Cash'] = self.starting_cash - 1000*world.iloc[0, world.columns.get_loc('Price')]
    baseline['Portfolio'] = baseline['Cash'] + baseline['Positions'] * world['Price']
    learner = self.learner
    
    
    start = self.calc_state(world, 0, 0)  # start with the new world at the start date and with no positions 
    
    s = start
    
    a = learner.test(self.calc_state(world, 0, 0)) # action is long, flat, short -- 0, 1, 2

    steps_remaining = world.shape[0] # Can only move forward in states up until end date
    day_count = 0
    trip_positions = []
    world['Cash'] = self.starting_cash
    world['Portfolio'] = world['Cash']
    world['Positions'] = 0
    trip_reward = 0
    # Each loop is one day
    while steps_remaining > 0:

      day = world.index[day_count]
  
      holdings = 0
      if a == 0:
        holdings = 1000
      if a == 1:
        holdings = 0
      if a == 2:
        holdings = -1000
  
    
      world.loc[day:, 'Positions'] = holdings
      holdings_change = world.iloc[day_count,  world.columns.get_loc('Positions')] - world.iloc[day_count-1,  world.columns.get_loc('Positions')] # Calculate change in position

      today_price = world.iloc[day_count, world.columns.get_loc('Price')]
  
      if holdings_change:
        
        world.loc[day:,'Cash'] =  world.loc[day,'Cash'] - (holdings_change * today_price) - abs(holdings_change * today_price)*self.floating_cost - self.fixed_cost 
        world.loc[day:,'Portfolio'] =  world.loc[day:, 'Positions'] * world.loc[day:, 'Price'] + world.loc[day:, 'Cash'] #Update today's portfolio
      
      # Apply the most recent action and determine its reward.
      s, r = self.query_world(world, day_count, s, a) # get a new state and a reward for our action 
     
      trip_positions.append(a)
      
      # Allow the learner to experience what happened.
      a = learner.test(self.calc_state(world, day_count, a))
      
      # Elapse time.
      steps_remaining -= 1
      day_count += 1
      
    
    print("Learner reward: ", world['Portfolio'][-1] - self.starting_cash)
    print("Baseline made: ", baseline['Portfolio'][-1] - self.starting_cash)
    
    learner_portfolio = world.loc[:,'Portfolio'] - self.starting_cash
    print(learner_portfolio)
    baseline_port = baseline['Portfolio'] - self.starting_cash
    data = {'Learner': learner_portfolio, 'Baseline': baseline_port}
    compare = pd.DataFrame(data)
    plot = compare.plot(title="QTrader vs. Baseline", colormap=cm.Accent)
    plot.grid()
    trades = world['Positions'].copy()
    trades.iloc[1:] = trades.diff().iloc[1:]
    trades.iloc[0] = 0
    trades = trades.to_frame()
    
    trades = trades.loc[(trades!=0).any(axis=1)]
    trades['Positions'] = trades['Positions'].cumsum()
    #print("Number of trades: ", trades.shape)
    
    for day in trades.index:
        if trades.loc[day,'Positions'] > 0:
            plt.axvline(x=day, color = 'blue', alpha=0.25) 
       
        elif trades.loc[day,'Positions'] < 0:
            plt.axvline(x=day, color = 'red', alpha=0.25) 
        else:
            plt.axvline(x=day, color = 'black', alpha=0.25)

    plt.show()
    
if __name__ == '__main__':
  # Load the requested stock for the requested dates, instantiate a Q-Learning agent,
  # and let it start trading.
  np.random.seed(759941)
  parser = argparse.ArgumentParser(description='Stock environment for Q-Learning.')

  date_args = parser.add_argument_group('date arguments')
  date_args.add_argument('--train_start', default='2018-01-01', metavar='DATE', help='Start of training period.')
  date_args.add_argument('--train_end', default='2019-12-31', metavar='DATE', help='End of training period.')
  date_args.add_argument('--test_start', default='2020-01-01', metavar='DATE', help='Start of testing period.')
  date_args.add_argument('--test_end', default='2021-12-31', metavar='DATE', help='End of testing period.')

  learn_args = parser.add_argument_group('learning arguments')
  learn_args.add_argument('--dyna', default=0, type=int, help='Dyna iterations per experience.')
  learn_args.add_argument('--eps', default=0.99, type=float, metavar='EPSILON', help='Starting epsilon for epsilon-greedy.')
  learn_args.add_argument('--eps_decay', default=0.99995, type=float, metavar='DECAY', help='Decay rate for epsilon-greedy.')

  sim_args = parser.add_argument_group('simulation arguments')
  sim_args.add_argument('--cash', default=200000, type=float, help='Starting cash for the agent.')
  sim_args.add_argument('--fixed', default=0.00, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.00', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--symbol', default='DIS', help='Stock symbol to trade.')
  sim_args.add_argument('--trips', default=500, type=int, help='Round trips through training data.')

  args = parser.parse_args()


  # Create an instance of the environment class.
  env = StockEnvironment(fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
                          share_limit = args.shares )

  # Construct, train, and store a Q-learning trader.
  env.train_learner( start = args.train_start, end = args.train_end,
                     symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                     eps = args.eps, eps_decay = args.eps_decay )

  # Test the learned policy and see how it does.
  
  # In sample.
  env.test_learner( start = args.train_start, end = args.train_end, symbol = args.symbol )

  # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
  #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )
  
