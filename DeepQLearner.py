# -*- coding: utf-8 -*-
"""
Created on Thu May  5 20:32:36 2022

@author: crawf
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:13:32 2022

@author: Stephen
Financial Machhine Learning 

https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc
"""

import numpy as np
import random 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras 
from collections import deque

LOSS= keras.losses.Huber()
OPTIMIZER = keras.optimizers.SGD() #stochastic gradient descent 
METRICS = ['accuracy']

class DeepQLearner:

    
    # need to incorporate epsilon greedy
    # we want to try learn policy without explicitly modeling the world (T, R)
    # Q(s, a) --> R maps every state-action pair to a real number that is the expected sum of current and future rewards from taking action a from state s
    # initalize q table randomly so always have some estimate of the q value for everry state-action pair 
    # q value must be summ of immediate reward for engaging a stateion-actiono pair and the discounted future rewards we expect to receive after taking that stateaction paiir 
    # policy(s) = argmaxQ[s, a] for all a from the state s; run until convergence i.e. q values do not change 
    
    
    """
    Construct class instance
    
    @param states: The number of distinct states 
    @param actions: The number of distinct actions 
    @param alpha: Learning rate
    @param gamme: Discount rate
    @param epsilon: Random action rate
    @param epsilon_decay: The rate at which the random action rate decreases after each random action
    """
    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9, epsilon = 0.98, epsilon_decay = 0.999, dyna=0):
        
        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.memory = deque(maxlen=1000) # store up to 1000 real experiences 
        self.states = states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna
        self.last_move = []
        self.actions = actions
        self.q_net = self.build_model() # a Q net to train
        self.target_net = self.build_model() # q net for evaluation 
        
    def build_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(units=self.states, input_dim=1, activation='relu'))
        model.add(layers.Dense(units=self.states/2, activation='relu'))
        model.add(layers.Dense(units=self.states/4, activation='relu'))
        model.add(layers.Dense(units=self.states/8, activation='relu'))
        model.add(layers.Dense(self.actions, activation='sigmoid'))
        model.compile(loss=self.loss, optimizer=OPTIMIZER, metrics=METRICS)
        print(model.summary())
        return model 
    
    
    """
    Define a custom loss function equal based on the predicted Q value and the correct Q value
    
    @param output_neuron: The value of the output neuron representing the predicted Q value
    @param correct value is the 'new estiamte from tabular'
    """
    def loss(self, predicted_value, correct_value):
        
        return tf.square(correct_value - predicted_value)

    
    def train (self, s, r):
        # Receive new state s and new reward r.  Update Q-table and return selected action.
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        #           How will you know the previous state and action?
            
        # grab the last state and action 
        
        last_experience = self.last_move
        
        old_s = last_experience[0]
        old_a = last_experience[1]
        
        # q-update 
        predicted_q_value = (self.q_net.predict(old_s)) 
        correct_q_value = (self.gamma*(self.q_net.predict(s)))
        self.model.fit(predicted_q_value, correct_q_value)
        
        # pick new action:
        epsilon_checker = random.random()
       
        # find the new action we plan to take from the new state
        if epsilon_checker < self.epsilon:
        
            a = random.randrange(0, self.actions)
            self.epsilon = self.epsilon * self.epsilon_decay
            
        else:
           a = 0 
           max_val = -1
           for i in range(self.actions):
               correct_q_value = (r + self.gamma*(self.q_net.predict(s, i)))
               if correct_q_value > max_val:
                   a = i
    
        self.experience_history.append([old_s, old_a, s, r])
        experience_index = np.random.randint(0, len(self.experience_history), (self.dyna,))
        
        for i in range(self.dyna): 
            
            exp = experience_index[i]
            selected_experience = self.experience_history[exp]

            dyna_s = selected_experience[0]
            dyna_a = selected_experience[1]
            dyna_s_prime = selected_experience[2]
            dyna_r = selected_experience[3]
            predicted_q_value = (self.q_net.predict(dyna_s)) 
            correct_q_value = (r + self.gamma*(self.q_net.predict(dyna_s)))
            self.model.fit(predicted_q_value, correct_q_value)
        
        self.last_move = [s, a]
         
        return a
    
    
    def test (self, s):
        # Receive new state s.  Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # When testing, you probably do not want to take random actions... (What good would it do?)
      
        s = np.array([s,])
        print("Getting first action from test state: ", s)
        action = self.q_net.test_on_batch(s)
        #action = (self.gamma*(self.q_net.predict(s)))
        print("After actions")
        self.last_move = [s, action]
        
        return action