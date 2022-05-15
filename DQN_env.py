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
from backtester_manual_trading import assess_strategy_dataframe
import itertools
 
TRIPS_WITHOUT_DYNA = 500
TRIPS_WITH_DYNA = 50
FAILURE_RATE = 0

class StockEnvironment:

  def __init__ (self, fixed = 0, floating = 0, starting_cash = None, share_limit = None):
    
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
    
    rsi = tech_ind.Relative_Strength_Index(price_data, 14)
    
    aroon = tech_ind.Aroon_Oscillator(price_data, 15)
    
    stochastic = tech_ind.Stochastic_Oscillator(price_data)
    
    bb = tech_ind.Bollinger_Bands(price_data, 14)
    
    obv = tech_ind.On_Balance_Volume(price_data, volume_data)
    
    world = price_data.copy()
    world.columns = ['Price']
    world['Williams Percent Range'] = williams['Williams Percentage']
    world['Bollinger Band Percentage'] = (world['Price'].sub(bb['SMA'], fill_value=np.nan))/(bb['Top Band'].sub(bb['Bottom Band'], fill_value=np.nan))
    world['OBV Normalized'] = obv/(abs(obv).rolling(window=5, min_periods=5).sum()) * 100 # scaled between -100 and 100
    world['RSI'] = rsi
    world['Aroon Up'] = aroon['Up']
    world['Aroon Down'] = aroon['Down']
    world['Stochastic'] = stochastic
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
  def calc_state (self, df, day, holdings, indicators):
    """ Quantizes the state to a single number. """
    state_list = [holdings] 
    
    for indicator in indicators:
         
        current_val = df.iloc[day, indicator+1] # 0 to -100 # 3 buckets 
        state_list.append(current_val)
  
    return state_list
  

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
                     eps = 0.0, eps_decay = 0.0, indicators = None):
    
    world = self.prepare_world(start, end, symbol)
    
    
    if dyna > 0:
      learner = DeepQLearner(states=len(indicators) + 1, actions = 3, epsilon=eps,epsilon_decay=eps_decay, dyna=dyna)
    else:
      learner = DeepQLearner(states=len(indicators) + 1, actions = 3, epsilon=eps,epsilon_decay=eps_decay)
    
    
    
    # Remember the total rewards of each trip individually.
    trip_rewards = []
    
    # Each loop is one trip through the state space 
    for i in range(trips):

      # A new trip starts with the learner at the start state with no rewards.
      # Get the initial action.
      world['Cash'] = self.starting_cash
      world['Portfolio'] = world['Cash']
      world['Positions'] = 0
      holdings = 0 
      trip_reward = 0
      
      # Each loop is one day
      for day_tracker in range(len(world)):
        
        day_key = world.index[day_tracker] # get the index key for the given day, i.e. '2018-01-01'
        
        if day_tracker == 0:
            a = learner.test(self.calc_state(world, day_tracker, holdings, indicators))
            cash_reward = 0
        else: 
            r = world.iloc[day_tracker, world.columns.get_loc('Portfolio')]/world.iloc[day_tracker - 1, world.columns.get_loc('Portfolio')] - 1
            a = learner.train(self.calc_state(world, day_tracker, holdings, indicators), r)
            cash_reward = world.iloc[day_tracker, world.columns.get_loc('Portfolio')] - world.iloc[day_tracker - 1, world.columns.get_loc('Portfolio')]
            
        if a == 0:
            holdings = 1000
        if a == 1:
            holdings = 0
        if a == 2:
            holdings = -1000
        
        world.loc[day_key:, 'Positions'] = holdings
        if day_tracker == 0:
            holdings_change = holdings
            
        else:
            holdings_change = world.diff().loc[day_key, 'Positions'] 
        
        today_price = world.iloc[day_tracker, world.columns.get_loc('Price')]
        
        if holdings_change:
            transaction_fees = (holdings_change * today_price) + abs(holdings_change)*(today_price)*self.floating_cost + self.fixed_cost    
            world.loc[day_key:,'Cash'] = world.loc[day_key,'Cash'] - transaction_fees
            world.loc[day_key:, 'Portfolio'] =  world.loc[day_key:, 'Positions'] * world.loc[day_key:, 'Price'] + world.loc[day_key:, 'Cash'] # Update today's portfolio
            
        day_tracker += 1
        # Allow the learner to experience what happened.
        trip_reward += cash_reward # The numeric value between yesterday's portfolio and today
        
      print("For trip ", i, " reward  is: ", trip_reward)
      trip_rewards.append(trip_reward)
        #Breakout when there is convergance (5 days in a row with same trip rewards)
      # if (i > 5 and trip_rewards[-1] == trip_rewards[-2] and trip_rewards[-2] == trip_rewards[-3] and trip_rewards[-3] == trip_rewards[-4] and trip_rewards[-4] == trip_rewards[-5]):
      #     break
     
    for i in range(len(trip_rewards)):
        print("For trip number ", i, " net result is: ", trip_rewards[i])
        
    self.learner = learner
    return learner


    
  def get_baseline(self, start=None, end=None, symbol=None):
     world = self.prepare_world(start, end, symbol)
     baseline = world['Price'].copy()
     baseline.iloc[:] = np.nan
     baseline.columns = ['Positions']
     baseline['Positions'] = 1000
     baseline['Cash'] = self.starting_cash - 1000*world.iloc[0, world.columns.get_loc('Price')]
     baseline['Portfolio'] = baseline['Cash'] + baseline['Positions'] * world['Price']
     baseline['Portfolio'] -= self.starting_cash
     return baseline 
    
    

  """
  A helper method for testing a learner that has already been trained.
  
  @param Start: Start Date
  @param End: End date
  @param Symbol: Stock symbol to trade
  """
  def test_learner( self, start = None, end = None, symbol = None, indicators = None):
    
    world = self.prepare_world(start, end, symbol)
    learner = self.learner
    
    world['Cash'] = self.starting_cash
    world['Portfolio'] = world['Cash']
    world['Positions'] = 0
    holdings = 0 
    trip_reward = 0
     

    # Each loop is one day
    for day_tracker in range(len(world)):
      
      day_key = world.index[day_tracker] # get the index key for the given day, i.e. '2018-01-01'
      
      state = self.calc_state(world, day_tracker, holdings, indicators)
      
      if day_tracker == 0:
          a = learner.test(state)
          cash_reward = 0
      else: 
          r = world.iloc[day_tracker, world.columns.get_loc('Portfolio')]/world.iloc[day_tracker - 1, world.columns.get_loc('Portfolio')] - 1
          a = learner.test(state)
          cash_reward = world.iloc[day_tracker, world.columns.get_loc('Portfolio')] - world.iloc[day_tracker - 1, world.columns.get_loc('Portfolio')]
          
      if a == 0:
          holdings = 1000
      if a == 1:
          holdings = 0
      if a == 2:
          holdings = -1000
      
      world.loc[day_key:, 'Positions'] = holdings
      if day_tracker == 0:
          holdings_change = holdings
          
      else:
          holdings_change = world.diff().loc[day_key, 'Positions'] 
      
      today_price = world.iloc[day_tracker, world.columns.get_loc('Price')]
      
      if holdings_change:
          transaction_fees = (holdings_change * today_price) + abs(holdings_change)*(today_price)*self.floating_cost + self.fixed_cost    
          world.loc[day_key:,'Cash'] = world.loc[day_key,'Cash'] - transaction_fees
          world.loc[day_key:, 'Portfolio'] =  world.loc[day_key:, 'Positions'] * world.loc[day_key:, 'Price'] + world.loc[day_key:, 'Cash'] # Update today's portfolio
          
      day_tracker += 1
      # Allow the learner to experience what happened.
      trip_reward += cash_reward # The numeric value between yesterday's portfolio and today
      
      
    
    print("Learner reward: ", world['Portfolio'][-1] - self.starting_cash)
    
    learner_portfolio = world.loc[:,'Portfolio'] - self.starting_cash
    indicators_list = ""
    for ind in indicators:
        indicators_list += world.columns[ind]
        indicators_list += " "
    print("Indicators list: ", indicators_list)
    learner_portfolio.rename(indicators_list)
    
    return learner_portfolio 
    
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
  sim_args.add_argument('--floating', default=0.00, type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--symbol', default='DIS', help='Stock symbol to trade.')
  sim_args.add_argument('--trips', default=1, type=int, help='Round trips through training data.')

  args = parser.parse_args()



    # Create an instance of the environment class.
env = StockEnvironment(fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
                       share_limit = args.shares)
    
baseline = env.get_baseline(start = args.test_start, end = args.test_end,
                   symbol = args.symbol)

solo_names = ['William Percent Range', 'Bollinger Band Percentage', 'OBV Normalized', 'RSI', 'Aroon Up', 'Aroon Down', 'Stochastic']

solo_comparisons = baseline['Portfolio']
solo_comparisons.name = 'Baseline'
print(solo_comparisons)
for i in range(7):
    indicator = [i]
    # Construct, train, and store a Q-lSearning trader.
    env.train_learner( start = args.train_start, end = args.train_end,
                        symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                        eps = args.eps, eps_decay = args.eps_decay, indicators=indicator)

    # Test the learned policy and see how it does.
    
    # In sample.
    port = env.test_learner(start = args.test_start, end = args.test_end, symbol = args.symbol, indicators=indicator)
    port.name =(solo_names[i])
    print(port)
    solo_comparisons =  pd.concat([solo_comparisons, port], axis=1)

    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )
name = '/Users/Stephen/Desktop/FML/One_Indicator_Results.csv'
print(solo_comparisons)
solo_comparisons.to_csv(name, index=True)
compare = pd.DataFrame(solo_comparisons['Baseline'])
solo_comparisons = solo_comparisons.drop(columns='Baseline')
print(solo_comparisons)
top_three = solo_comparisons.iloc[-1, np.argsort(-solo_comparisons.values[0])[:3]]
print(top_three)
for i in top_three.index:
    compare = pd.concat([compare, solo_comparisons[i]], axis=1)

name = '/Users/Stephen/Desktop/FML/One_Indicator_Results.pdf'
compare.plot.line(title='One Indicator Results', colormap=cm.Accent)
plt.savefig(name)

abr_names = ['WPR', 'BBP', 'OBV', 'RSI', 'AU', 'AD', 'S']

seven_comparisons = baseline['Portfolio']
seven_comparisons.name = 'Baseline'
for combo in itertools.combinations(range(7), 7):
    # Create an instance of the environment class.
    
    # Construct, train, and store a Q-learning trader.
    env.train_learner( start = args.train_start, end = args.train_end,
                        symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                        eps = args.eps, eps_decay = args.eps_decay, indicators=combo)

    # Test the learned policy and see how it does.
    
    # In sample.
    port = env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol, indicators=combo)
    combo_name = ""
    for c in combo:
        combo_name += abr_names[c]
        combo_name += " & "
    port.name = combo_name[:-3]
    seven_comparisons = pd.concat([seven_comparisons, port], axis=1)

    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol 
    
print(seven_comparisons)
name = '/Users/Stephen/Desktop/FML/Seven_Indicator_Results.csv'
seven_comparisons.to_csv(name, index=True)
compare = pd.DataFrame(seven_comparisons['Baseline'])
seven_comparisons = seven_comparisons.drop(columns='Baseline')
print(seven_comparisons)
top_three = seven_comparisons.iloc[-1, np.argsort(-seven_comparisons.values[0])[:3]]
print(top_three)
for i in top_three.index:
    compare = pd.concat([compare, seven_comparisons[i]], axis=1)
    
name = '/Users/Stephen/Desktop/FML/Seven_Indicator_Results.pdf'
compare.plot.line(title='Seven Indicator Results', colormap=cm.Accent, alpha=0.5)
plt.savefig(name)


six_comparisons = baseline['Portfolio']
six_comparisons.name = 'Baseline'
for combo in itertools.combinations(range(7), 6):
    # Create an instance of the environment class.
    
    # Construct, train, and store a Q-learning trader.
    env.train_learner( start = args.train_start, end = args.train_end,
                       symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                       eps = args.eps, eps_decay = args.eps_decay, indicators=combo)

    # Test the learned policy and see how it does.
    
    # In sample.
    port = env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol, indicators=combo)
    combo_name = ""
    for c in combo:
        combo_name += abr_names[c]
        combo_name += " & "
    port.name = combo_name[:-3]
    six_comparisons = pd.concat([six_comparisons, port], axis=1)

    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol 
    
print(six_comparisons)
name = '/Users/Stephen/Desktop/FML/Six_Indicator_Results.csv'
six_comparisons.to_csv(name, index=True)
compare = pd.DataFrame(six_comparisons['Baseline'])
six_comparisons = six_comparisons.drop(columns='Baseline')
print(six_comparisons)
top_three = six_comparisons.iloc[-1, np.argsort(-six_comparisons.values[0])[:3]]
print(top_three)
for i in top_three.index:
    compare = pd.concat([compare, six_comparisons[i]], axis=1)

name = '/Users/Stephen/Desktop/FML/Six_Indicator_Results.pdf'
compare.plot.line(title='Six Indicator Results', colormap=cm.Accent, alpha=0.5)
plt.savefig(name)


five_comparisons = baseline['Portfolio']
five_comparisons.name = 'Baseline'
for combo in itertools.combinations(range(7), 5):
    # Create an instance of the environment class.
    
    # Construct, train, and store a Q-learning trader.
    env.train_learner( start = args.train_start, end = args.train_end,
                       symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                       eps = args.eps, eps_decay = args.eps_decay, indicators=combo)

    # Test the learned policy and see how it does.
    
    # In sample.
    port = env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol, indicators=combo)
    combo_name = ""
    for c in combo:
        combo_name += abr_names[c]
        combo_name += " & "
    port.name = combo_name[:-3]
    five_comparisons = pd.concat([five_comparisons, port], axis=1)

    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol 
    
print(five_comparisons)
name = '/Users/Stephen/Desktop/FML/Five_Indicator_Results.csv'
five_comparisons.to_csv(name, index=True)
compare = pd.DataFrame(five_comparisons['Baseline'])
five_comparisons = five_comparisons.drop(columns='Baseline')
print(five_comparisons)
top_three = five_comparisons.iloc[-1, np.argsort(-five_comparisons.values[0])[:3]]
print(top_three)
for i in top_three.index:
    compare = pd.concat([compare, five_comparisons[i]], axis=1)

name = '/Users/Stephen/Desktop/FML/Five_Indicator_Results.pdf'
compare.plot.line(title='Five Indicator Results', colormap=cm.Accent, alpha=0.5)
plt.savefig(name)


four_comparisons = baseline['Portfolio']
four_comparisons.name = 'Baseline'
for combo in itertools.combinations(range(7), 4):
    # Create an instance of the environment class.
    
    # Construct, train, and store a Q-learning trader.
    env.train_learner( start = args.train_start, end = args.train_end,
                       symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                       eps = args.eps, eps_decay = args.eps_decay, indicators=combo)

    # Test the learned policy and see how it does.
    
    # In sample.
    port = env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol, indicators=combo)
    combo_name = ""
    for c in combo:
        combo_name += abr_names[c]
        combo_name += " & "
    port.name = combo_name[:-3]
    four_comparisons = pd.concat([four_comparisons, port], axis=1)

    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol 
    
print(four_comparisons)
name = '/Users/Stephen/Desktop/FML/Four_Indicator_Results.csv'
four_comparisons.to_csv(name, index=True)
compare = pd.DataFrame(four_comparisons['Baseline'])
four_comparisons = four_comparisons.drop(columns='Baseline')
print(four_comparisons)
top_three = four_comparisons.iloc[-1, np.argsort(-four_comparisons.values[0])[:3]]
print(top_three)
for i in top_three.index:
    compare = pd.concat([compare, four_comparisons[i]], axis=1)
    

name = '/Users/Stephen/Desktop/FML/Four_Indicator_Results.pdf'
compare.plot.line(title='Four Indicator Results', colormap=cm.Accent, alpha=0.5)
plt.savefig(name)


three_comparisons = baseline['Portfolio']
three_comparisons.name = 'Baseline'
for combo in itertools.combinations(range(7), 3):
    # Create an instance of the environment class.
    
    # Construct, train, and store a Q-learning trader.
    env.train_learner( start = args.train_start, end = args.train_end,
                       symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                       eps = args.eps, eps_decay = args.eps_decay, indicators=combo)

    # Test the learned policy and see how it does.
    
    # In sample.
    port = env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol, indicators=combo)
    combo_name = ""
    for c in combo:
        combo_name += abr_names[c]
        combo_name += " & "
    port.name = combo_name[:-3]
    three_comparisons = pd.concat([three_comparisons, port], axis=1)

    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol 
    
print(three_comparisons)
name = '/Users/Stephen/Desktop/FML/Three_Indicator_Results.csv'
three_comparisons.to_csv(name, index=True)
compare = pd.DataFrame(three_comparisons['Baseline'])
three_comparisons = three_comparisons.drop(columns='Baseline')
print(three_comparisons)
top_three = three_comparisons.iloc[-1, np.argsort(-three_comparisons.values[0])[:3]]
print(top_three)
for i in top_three.index:
    compare = pd.concat([compare, three_comparisons[i]], axis=1)
    

name = '/Users/Stephen/Desktop/FML/Three_Indicator_Results.pdf'
compare.plot.line(title='Three Indicator Results', colormap=cm.Accent, alpha=0.5)
plt.savefig(name)


two_comparisons = baseline['Portfolio']
two_comparisons.name = 'Baseline'
for combo in itertools.combinations(range(7), 2):
    # Create an instance of the environment class.
    
    # Construct, train, and store a Q-learning trader.
    env.train_learner( start = args.train_start, end = args.train_end,
                       symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                       eps = args.eps, eps_decay = args.eps_decay, indicators=combo)

    # Test the learned policy and see how it does.
    
    # In sample.
    port = env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol, indicators=combo)
    combo_name = ""
    for c in combo:
        combo_name += abr_names[c]
        combo_name += " & "
    port.name = combo_name[:-3]
    seven_comparisons = pd.concat([two_comparisons, port], axis=1)

    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol 
    
print(two_comparisons)
name = '/Users/Stephen/Desktop/FML/Two_Indicator_Results.csv'
two_comparisons.to_csv(name, index=True)
compare = pd.DataFrame(two_comparisons['Baseline'])
two_comparisons = two_comparisons.drop(columns='Baseline')
print(seven_comparisons)
top_three = two_comparisons.iloc[-1, np.argsort(-two_comparisons.values[0])[:3]]
print(top_three)
for i in top_three.index:
    compare = pd.concat([compare, two_comparisons[i]], axis=1)

name = '/Users/Stephen/Desktop/FML/Two_Indicator_Results.pdf'
compare.plot.line(title='Two Indicator Results', colormap=cm.Accent, alpha=0.5)
plt.savefig(name)
