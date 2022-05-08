#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Do not change your TabularQLearner.py code.  It contains generic Q-Learning that does not care what problem it solves.
Use exactly the same indicators as in the prior technical strategy project to produce comparable results.
In sample testing should be performed by calling your test() method with the same dates used to train.
Out of sample testing should be performed by calling your test() method with dates in the future of those used to train.
Only perform out of sample testing at the very end, once you are satisfied with your in sample results.

State should be (day, shares, indicator_1, indicator_2, indicator_3)
"""
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  
from DeepQLearner import DeepQLearner
import tech_ind 
import sys
import timeit
import datetime 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TRIPS_WITHOUT_DYNA = 500
TRIPS_WITH_DYNA = 50
FAILURE_RATE = 0

class DQNStockEnvironment:

    def __init__ (self, fixed = None, floating = None, starting_cash = None, share_limit = None):

        self.shares = share_limit
        self.fixed_cost = fixed
        self.floating_cost = floating
        self.starting_cash = starting_cash
        self.learner = None

    def prepare_world (self, start_date, end_date, symbol):
        """
        Read the relevant price data and calculate some indicators.
        Return a DataFrame containing everything you need.
        """
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


    def calc_state (self, df, day, holdings):
        """ Quantizes the state to a single number. """


        world = df
        #print("World is: ", world)

        current_williams = df.iloc[day]['Williams Percent Range'] # 0 to -100 # 3 buckets 
        current_bbp = df.iloc[day]['Bollinger Band Percentage'] # x < 0 < x < sma < x < top band < x <--buckets 0, 1, 2, 3
        current_obv = df.iloc[day]['OBV Normalized'] 
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

        #print("Holdings: ", holdings, " Current  Williams: ", current_williams, " current_bbp: ", current_bbp, " current obv: ", current_obv, " STATE: ", state)
        return int(state)


    def query_world(self, world, day, s, a, prev_a):
        # Apply the action to the state and return the new state and reward.

        if np.random.random() < FAILURE_RATE: # what if we cannot make a perfect trade everytime? 
            a = np.random.randint(3) 

        holdings = 0

        if a == 0:
            holdings = 1000
        if a == 1:
            holdings = 0
        if a == 2:
            holdings = -1000

        prev_hold = 0
        if prev_a == 0:
            prev_hold = 1000
        if prev_a == 1:
            prev_hold = 0
        if prev_a == 2:
            prev_hold = -1000        

        day_next = day + 1

        if day == world.shape[0] - 1:
            day_next = day

        s_prime = self.calc_state(world, day_next, holdings)

        daily_returns = world.iloc[day]['Price'] - world.iloc[day-1]['Price']
        r = (world.iloc[day]['Price']/ world.iloc[day-1]['Price']) * holdings

        return s_prime, r


    def train_learner( self, start = None, end = None, symbol = None, trips = 0, dyna = 0,
                     eps = 0.0, eps_decay = 0.0):
        """
        Construct a Q-Learning trader and train it through many iterations of a stock
        world.  Store the trained learner in an instance variable for testing.

        Print a summary result of what happened at the end of each trip.
        Feel free to include portfolio stats or other information, but AT LEAST:

        Trip 499 net result: $13600.00
        """

        world = self.prepare_world(start, end, symbol)

        world_size = 144 # 3 * 3 * 4 * 4

        if dyna > 0:
            learner = DeepQLearner(states=world_size, actions = 3, epsilon=eps,epsilon_decay=eps_decay, dyna=dyna)
        else:
            learner = DeepQLearner(states=world_size, actions = 3, epsilon=eps,epsilon_decay=eps_decay)

        start = self.calc_state(world, 0, 0)  # start with the new world at the start date and with no positions 

        #goal = [self.calc_state(world, -1, 0), self.calc_state(world, -1, 1000)]   # two fair options that are acceptable end states -- do not want to allow shorting at time end 

        # Remember the total rewards of each trip individually.
        trip_rewards = []

        # Each loop is one trip through the state space 
        for i in range(trips):

            # A new trip starts with the learner at the start state with no rewards.
            # Get the initial action.

            s = start
            trip_reward = 0
            a = learner.test(self.calc_state(world, 0, 0)) # action is long, flat, short -- 0, 1, 2

            steps_remaining = world.shape[0] # Can only move forward in states up until end date

            day_count = 0
            prev_a = 0

            # Each loop is one day

            while steps_remaining > 0:


                # Apply the most recent action and determine its reward.
                s, r = self.query_world(world, day_count, s, a, prev_a) # get a new state and a reward for our action 

                # Allow the learner to experience what happened.
                a = learner.train(self.calc_state(world, day_count, a), r)

                # Accumulate the total rewards for this trip.
                trip_reward += r

                # Elapse time.
                steps_remaining -= 1
                day_count += 1

            # Remember the total reward of each trip.

            trip_rewards.append(trip_reward)

            print("For trip number ", i, " net result is: ", trip_rewards[i])


        #for i in range(len(trip_rewards)):
            #   print("For trip number ", i, " net result is: ", trip_rewards[i])

        self.learner = learner
        return learner



    def test_learner( self, start = None, end = None, symbol = None):
        """
        Evaluate a trained Q-Learner on a particular stock trading task.
        Print a summary result of what happened during the test.

        Feel free to include portfolio stats or other information, but AT LEAST:

        Test trip, net result: $31710.00
        Benchmark result: $6690.0000
        """

        world = self.prepare_world(start, end, symbol)
        baseline = world['Price'].copy()
        baseline.iloc[:] = np.nan
        baseline.columns = ['Positions']
        baseline['Positions'] = 1000
        baseline['Cash'] = 100000 - 1000*world['Price'][0]
        baseline['Portfolio'] = baseline['Cash'] + baseline['Positions'] * world['Price']
        learner = self.learner


        start = self.calc_state(world, 0, 0)  # start with the new world at the start date and with no positions 

        s = start
        trip_reward = 0

        a = learner.test(self.calc_state(world, 0, 0)) # action is long, flat, short -- 0, 1, 2

        steps_remaining = world.shape[0] # Can only move forward in states up until end date
        day_count = 0
        trip_positions = []

        # Each loop is one day
        while steps_remaining > 0:

            # Apply the most recent action and determine its reward.
            s, r = self.query_world(world, day_count, s, a) # get a new state and a reward for our action 

            trip_positions.append(a)

            # Allow the learner to experience what happened.
            a = learner.test(self.calc_state(world, day_count, a))

            # Accumulate the total rewards for this trip.
            trip_reward += r

            # Elapse time.
            steps_remaining -= 1
            day_count += 1

        # Remember the total reward of each trip.

        print("Learner reward: ", trip_reward)
        print("Baseline made: ", baseline['Portfolio'][-1] - 100000)


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
    env = DQNStockEnvironment(fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
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