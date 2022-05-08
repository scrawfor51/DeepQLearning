import numpy as np
import random 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#stock_env = stock_env.StockEnvironment()
TRAIN_LENGTH = 300
TEST_LENGTH = 100
LOSS = keras.losses.Huber()
OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)
METRICS = ['accuracy']
# Using huber loss for stability
loss_function = keras.losses.Huber()

class DeepQLearner:
    def __init__ (self, states = 100, actions = 3, alpha = 0.2, gamma = 0.9, epsilon = 0.98, epsilon_decay = 0.999, dyna=0, model_name=""):
        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        
        np.random.seed(759941)
        #self.experience_history = deque(maxlen=1000) # storeone thousand past expereinces for replay
        self.optimizer=keras.optimizers.Adam(lr=0.001)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.states = states
        self.actions = actions
        self.q_net = keras.load_model(model_name) if len(model_name) else self.build_model()
        self.q_net_target = None if len(model_name) else self.build_model()         
        
        #Experience replay buffers
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = [] 
        self.prev_state = None
        self.prev_action = None
        
    def build_model(self):
        #Using https://stackoverflow.com/questions/69933345/expected-min-ndim-2-found-ndim-1-full-shape-received-none
        single_feature_normalizer = tf.keras.layers.Normalization(axis=None)
        feature = tf.random.normal((self.states, 1))
        single_feature_normalizer.adapt(feature)
        
        model = keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            single_feature_normalizer,
            tf.keras.layers.Dense(1)
        ])
        model.add(layers.Dense(units=self.states, input_dim=self.states, activation="relu"))
        model.add(layers.Dense(units=self.states/2, activation="relu"))
        model.add(layers.Dense(units=self.states/4, activation="relu"))
        model.add(layers.Dense(self.actions/8, activation="relu"))
        model.add(layers.Dense(self.actions, activation="sigmoid"))
        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
        return model    
        

    def train (self, s, r):
        # Receive new state s and new reward r.  Update Q-table and return selected action.
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        #           How will you know the previous state and action?
        
        #Batch size??
        batch_size = 4
        
        #Take a random action
        if (np.random.random() < self.epsilon):
            a = random.randint(0, self.actions - 1)
            if self.prev_action is None:
                return a
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(s)
            state_tensor = tf.expand_dims(s, 0)        
            action_probs = self.q_net(state_tensor, training=False)
            # Take best action
            a = tf.argmax(action_probs[0]).numpy()
            
        
        #Decay epsilon
        self.epsilon *= self.epsilon_decay
        
        # Save actions and states in replay buffer        
        self.action_history.append(self.prev_action)
        self.state_history.append(self.prev_state)
        self.state_next_history.append(s)
        self.rewards_history.append(r)
        
        #Only update when above batchsize
        if len(self.rewards_history) > batch_size:
        
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(self.rewards_history)), size=batch_size)
            
            # Using list comprehension to sample from replay buffer
            state_sample = np.array([self.state_history[i] for i in indices])
            state_next_sample = np.array([self.state_next_history[i] for i in indices])
            rewards_sample = [self.rewards_history[i] for i in indices]
            action_sample = [self.action_history[i] for i in indices]
                
            future_rewards = self.q_net.predict(state_next_sample)        
            
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
            
            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, self.actions)
    
            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.q_net(state_sample)
    
                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)
    
            # Backpropagation
            grads = tape.gradient(loss, self.q_net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        
        
        self.prev_action = a
        self.prev_state = s
        return a

    def test (self, s):
        # Receive new state s.  Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # When testing, you probably do not want to take random actions... (What good would it do?)
        
        
        state_tensor = tf.convert_to_tensor(s)
        state_tensor = tf.expand_dims(s, 0)        
        action_probs = self.q_net(state_tensor, training=False)
        a = tf.argmax(action_probs[0]).numpy()
        
        
        self.prev_state = s
        self.prev_action = a
        
        return a