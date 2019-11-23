import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import random
import time
import tensorflow as tf
from tqdm import tqdm
from keras.optimizers import Adam
from bgu_utils import moving_average


# Agent class

REPLAY_MEMORY_SIZE = 777  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 64  # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 10  # The C value
MIN_EPSILON = 0.01
_LR = 7e-4


class DQNAgent:
    def __init__(self, env, hidden_layers=3, minibatch_size=64, discount=0.99, is_ddqn=False, use_tensorboard=False, weights=None):
        self.env = env
        self.use_tb = use_tensorboard
        self._is_ddqn = is_ddqn
        self._weights = weights
        if self.use_tb:
            self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}".format(int(time.time())))
        self._minibatch_size = minibatch_size
        self._discount = discount
        self._hidden_layers = hidden_layers
        self._observation_space_size = len(self.env.observation_space.bounded_below)
        self._action_space_size = self.env.action_space.n
        
        # Main model
        self.model = self._create_model()

        # Target network
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        
    def _create_model(self, units=32):
        model = Sequential()
        model.add(Dense(units=units, activation='relu', input_dim=self._observation_space_size))
        for _ in range(self._hidden_layers - 1):
            model.add(Dense(units=units, activation='relu'))
            
        model.add(Dense(units=self._action_space_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=_LR))
        if self._weights is not None:
            model.load_weights(self._weights)
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    # Trains main network every step during episode
    def _train_on_minibatch(self, is_terminal):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self._minibatch_size)

        y = []
        X = []
        
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            
            current_qs = self.get_qs(current_state)
            
            if not done:
                future_qs = self.target_model.predict(next_state[None])[0]
                
                if self._is_ddqn:
                    future_action = np.argmax(self.get_qs(next_state))
                    max_future_q = future_qs[future_action]
                else:
                    max_future_q = np.max(future_qs)
                    
                new_q = reward + self._discount * max_future_q
            else:
                new_q = reward
                
            # Update Q value for given state
            current_qs[action] = new_q
            
            y.append(current_qs)
            X.append(current_state)
            
        X = np.array(X)
        y = np.array(y)
        
        self.model.fit(X, y, batch_size=self._minibatch_size, verbose=0, shuffle=False, 
                       callbacks=[self.tensorboard] if self.use_tb and is_terminal else None)

        self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
            
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state[None])[0]
    
    def train(self, episodes, epsilon, epsilon_decay, show_every=500):
        # Iterate over episodes
        ep_rewards = []
        ep_avg_loss = np.zeros(episodes)
        for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes'):
            # Update tensorboard step every episode
            if self.use_tb:
                self.tensorboard.step = episode
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            # Reset environment and get initial state
            current_state = self.env.reset()
            # Reset flag and start iterating until episode ends
            done = False
            loss = []
            while not done:
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(self.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, self._action_space_size)

                new_state, reward, done, _ = self.env.step(action)
                if done and episode_reward < 499:
                    reward = 0  # cart-pole bug
                    
                # Transform new continous state to new discrete state and count reward
                
                episode_reward += reward
                if not episode % show_every:
                    self.env.render()

                # Every step we update replay memory and train main network
                self.update_replay_memory((current_state, action, reward, new_state, done))
                self._train_on_minibatch(done)
                current_state = new_state
                
            self.env.close()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            
            if self.use_tb:  # log episode total reward to tensorboard
                self.tensorboard.update_stats(reward=episode_reward, 
                                              epsilon=epsilon, 
                                              moving_average_reward=np.mean(ep_rewards[-100:]))

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= epsilon_decay
                epsilon = max(MIN_EPSILON, epsilon)
                
        self.model.save_weights(f"dqn{self._hidden_layers}.h5")
        return np.array(ep_rewards), ep_avg_loss

    def simulate(self):
        episode_reward = 0
        # Reset environment and get initial state
        current_state = self.env.reset()
        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            action = np.argmax(self.get_qs(current_state))
            new_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            self.env.render()
            current_state = new_state
        self.env.close()
        return episode_reward

        
# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

