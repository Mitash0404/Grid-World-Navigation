"""
Deep Q-Learning (DQN) Implementation

Implements Deep Q-Network using TensorFlow/Keras for grid world navigation.
Includes experience replay, target network, and comprehensive optimization.

Author: Mitash Shah
Date: May 2024 - Aug 2024
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import time


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    
    Features:
    - Neural network approximation of Q-function
    - Experience replay buffer
    - Target network for stable training
    - Comprehensive hyperparameter tuning
    - Performance optimization and metrics
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.001, discount_factor: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 2000,
                 batch_size: int = 32, target_update_freq: int = 10):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Number of possible states
            action_size: Number of possible actions
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            memory_size: Size of experience replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self._update_target_network()
        
        # Performance tracking
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'epsilon_values': [],
            'success_rate': [],
            'loss': []
        }
        
        # Training metrics
        self.episode_count = 0
        self.success_count = 0
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build the neural network model.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _update_target_network(self) -> None:
        """Update target network with current Q-network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state: int, action: int, reward: float, 
                 next_state: int, done: bool) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: int, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy strategy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action to take
        """
        if training and random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            # Convert state to one-hot encoding for neural network
            state_vector = self._state_to_vector(state)
            q_values = self.q_network.predict(state_vector, verbose=0)
            return np.argmax(q_values[0])
    
    def _state_to_vector(self, state: int) -> np.ndarray:
        """Convert state index to one-hot vector."""
        state_vector = np.zeros(self.state_size)
        state_vector[state] = 1.0
        return state_vector.reshape(1, -1)
    
    def replay(self) -> None:
        """Train the neural network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare training data
        states = np.array([self._state_to_vector(exp[0])[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([self._state_to_vector(exp[3])[0] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Target Q-values
        next_q_values = self.target_network.predict(next_states, verbose=0)
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * np.max(next_q_values[i])
        
        # Train the model
        history = self.q_network.fit(states, target_q_values, epochs=1, verbose=0)
        
        # Track loss
        if 'loss' in history.history:
            self.training_history['loss'].append(history.history['loss'][0])
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, episodes: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the DQN agent.
        
        Args:
            env: Environment to train on
            episodes: Number of training episodes
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        start_time = time.time()
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                action = self.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train on batch
                self.replay()
            
            # Track performance
            if done and info.get('position') == env.goal_position:
                self.success_count += 1
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(steps)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['success_rate'].append(self.success_count / (episode + 1))
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Update target network
            if episode % self.target_update_freq == 0:
                self._update_target_network()
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                success_rate = self.success_count / (episode + 1)
                avg_reward = np.mean(self.training_history['rewards'][-100:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Success Rate: {success_rate:.3f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        return {
            'total_episodes': episodes,
            'success_rate': self.success_count / episodes,
            'avg_reward': np.mean(self.training_history['rewards']),
            'avg_steps': np.mean(self.training_history['steps']),
            'training_time': training_time,
            'final_epsilon': self.epsilon
        }
    
    def evaluate(self, env, episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment to evaluate on
            episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        rewards = []
        steps = []
        success_count = 0
        
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < 1000:
                action = self.act(state, training=False)
                state, reward, done, info = env.step(action)
                total_reward += reward
                step_count += 1
            
            rewards.append(total_reward)
            steps.append(step_count)
            
            if done and info.get('position') == env.goal_position:
                success_count += 1
        
        return {
            'success_rate': success_count / episodes,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(steps),
            'std_steps': np.std(steps)
        }
    
    def plot_training_progress(self) -> None:
        """Plot training progress metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Rewards over time
        axes[0, 0].plot(self.training_history['rewards'])
        axes[0, 0].set_title('Rewards per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Success rate over time
        axes[0, 1].plot(self.training_history['success_rate'])
        axes[0, 1].set_title('Success Rate Over Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        
        # Steps per episode
        axes[0, 2].plot(self.training_history['steps'])
        axes[0, 2].set_title('Steps per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        
        # Epsilon decay
        axes[1, 0].plot(self.training_history['epsilon_values'])
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        
        # Loss over time
        if self.training_history['loss']:
            axes[1, 1].plot(self.training_history['loss'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
        
        # Moving average of rewards
        if len(self.training_history['rewards']) > 50:
            window = 50
            moving_avg = np.convolve(self.training_history['rewards'], 
                                   np.ones(window)/window, mode='valid')
            axes[1, 2].plot(moving_avg)
            axes[1, 2].set_title(f'Moving Average Rewards (window={window})')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Moving Average Reward')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and parameters."""
        # Save neural network
        self.q_network.save(f"{filepath}_q_network.h5")
        self.target_network.save(f"{filepath}_target_network.h5")
        
        # Save parameters and history
        model_data = {
            'parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            },
            'training_history': self.training_history
        }
        
        with open(f"{filepath}_params.json", 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model and parameters."""
        # Load neural networks
        self.q_network = tf.keras.models.load_model(f"{filepath}_q_network.h5")
        self.target_network = tf.keras.models.load_model(f"{filepath}_target_network.h5")
        
        # Load parameters
        with open(f"{filepath}_params.json", 'r') as f:
            model_data = json.load(f)
        
        params = model_data['parameters']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        self.memory_size = params['memory_size']
        self.batch_size = params['batch_size']
        self.target_update_freq = params['target_update_freq']
        self.training_history = model_data['training_history']


class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN agent with separate value and advantage streams.
    
    Improves learning by separating state value and action advantage estimation.
    """
    
    def _build_model(self) -> tf.keras.Model:
        """Build dueling DQN architecture."""
        input_layer = Input(shape=(self.state_size,))
        
        # Shared feature extraction
        shared = Dense(128, activation='relu')(input_layer)
        shared = Dense(128, activation='relu')(shared)
        shared = Dense(64, activation='relu')(shared)
        
        # Value stream
        value_stream = Dense(32, activation='relu')(shared)
        value = Dense(1, activation='linear')(value_stream)
        
        # Advantage stream
        advantage_stream = Dense(32, activation='relu')(shared)
        advantage = Dense(self.action_size, activation='linear')(advantage_stream)
        
        # Combine value and advantage
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        model = tf.keras.Model(inputs=input_layer, outputs=q_values)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model


class PrioritizedReplayDQNAgent(DQNAgent):
    """
    DQN agent with prioritized experience replay.
    
    Samples experiences based on their TD error for more efficient learning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priorities = deque(maxlen=self.memory_size)
        self.alpha = 0.6  # Prioritization exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001
    
    def remember(self, state: int, action: int, reward: float, 
                 next_state: int, done: bool, td_error: float = 1.0) -> None:
        """Store experience with priority."""
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(abs(td_error) + 1e-6)  # Add small constant to avoid zero priority
    
    def _get_priorities(self) -> np.ndarray:
        """Get current priorities for sampling."""
        return np.array(list(self.priorities))
    
    def _sample_batch(self) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with prioritized experience replay."""
        if len(self.memory) < self.batch_size:
            return random.sample(self.memory, len(self.memory)), None, None
        
        priorities = self._get_priorities()
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        batch = [self.memory[i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return batch, indices, weights
    
    def replay(self) -> None:
        """Train with prioritized experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        batch, indices, weights = self._sample_batch()
        
        # Prepare training data
        states = np.array([self._state_to_vector(exp[0])[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([self._state_to_vector(exp[3])[0] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Target Q-values
        next_q_values = self.target_network.predict(next_states, verbose=0)
        target_q_values = current_q_values.copy()
        
        # Calculate TD errors for priority updates
        td_errors = []
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
                td_error = abs(rewards[i] - current_q_values[i][actions[i]])
            else:
                target = rewards[i] + self.discount_factor * np.max(next_q_values[i])
                target_q_values[i][actions[i]] = target
                td_error = abs(target - current_q_values[i][actions[i]])
            
            td_errors.append(td_error)
        
        # Update priorities
        for i, idx in enumerate(indices):
            self.priorities[idx] = abs(td_errors[i]) + 1e-6
        
        # Train the model with importance sampling weights
        history = self.q_network.fit(states, target_q_values, 
                                   sample_weight=weights, epochs=1, verbose=0)
        
        # Track loss
        if 'loss' in history.history:
            self.training_history['loss'].append(history.history['loss'][0])
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

