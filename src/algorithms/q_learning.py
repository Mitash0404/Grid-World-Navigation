"""
Q-Learning Algorithm Implementation

Implements Q-Learning, SARSA, and Double Q-Learning algorithms for grid world navigation.
Includes comprehensive parameter tuning and performance optimization.

Author: Mitash Shah
Date: May 2024 - Aug 2024
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt
from collections import deque
import json
import time


class QLearningAgent:
    """
    Q-Learning agent with comprehensive parameter tuning and optimization.
    
    Features:
    - Standard Q-Learning implementation
    - Epsilon-greedy exploration strategy
    - Learning rate and discount factor optimization
    - Performance tracking and metrics
    - Model saving and loading
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.99,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_size: Number of possible states
            action_size: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        
        # Performance tracking
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'epsilon_values': [],
            'success_rate': []
        }
        
        # Parameter optimization tracking
        self.parameter_history = []
    
    def act(self, state: int, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy strategy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action to take
        """
        if training and random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """
        Update Q-table using Q-Learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Maximum Q-value for next state
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-Learning update
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, episodes: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the agent using Q-Learning.
        
        Args:
            env: Environment to train on
            episodes: Number of training episodes
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        start_time = time.time()
        success_count = 0
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:  # Prevent infinite episodes
                action = self.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Track performance
            if done and info.get('position') == env.goal_position:
                success_count += 1
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(steps)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['success_rate'].append(success_count / (episode + 1))
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                success_rate = success_count / (episode + 1)
                avg_reward = np.mean(self.training_history['rewards'][-100:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Success Rate: {success_rate:.3f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        return {
            'total_episodes': episodes,
            'success_rate': success_count / episodes,
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        axes[1, 0].plot(self.training_history['steps'])
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        # Epsilon decay
        axes[1, 1].plot(self.training_history['epsilon_values'])
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained Q-table and parameters."""
        model_data = {
            'q_table': self.q_table.tolist(),
            'parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            },
            'training_history': self.training_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained Q-table and parameters."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.q_table = np.array(model_data['q_table'])
        params = model_data['parameters']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        self.training_history = model_data['training_history']


class SARSAAgent(QLearningAgent):
    """
    SARSA (State-Action-Reward-State-Action) agent.
    
    On-policy learning algorithm that uses the actual action taken
    in the next state for Q-value updates.
    """
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, next_action: int, done: bool) -> None:
        """
        Update Q-table using SARSA update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action taken
            done: Whether episode is finished
        """
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * self.q_table[next_state, next_action]
        
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def train(self, env, episodes: int, verbose: bool = True) -> Dict[str, Any]:
        """Train the agent using SARSA."""
        start_time = time.time()
        success_count = 0
        
        for episode in range(episodes):
            state = env.reset()
            action = self.act(state, training=True)
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                next_state, reward, done, info = env.step(action)
                next_action = self.act(next_state, training=True)
                
                self.update(state, action, reward, next_state, next_action, done)
                
                state, action = next_state, next_action
                total_reward += reward
                steps += 1
            
            if done and info.get('position') == env.goal_position:
                success_count += 1
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(steps)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['success_rate'].append(success_count / (episode + 1))
            
            self.decay_epsilon()
            
            if verbose and (episode + 1) % 100 == 0:
                success_rate = success_count / (episode + 1)
                avg_reward = np.mean(self.training_history['rewards'][-100:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Success Rate: {success_rate:.3f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        return {
            'total_episodes': episodes,
            'success_rate': success_count / episodes,
            'avg_reward': np.mean(self.training_history['rewards']),
            'avg_steps': np.mean(self.training_history['steps']),
            'training_time': training_time,
            'final_epsilon': self.epsilon
        }


class DoubleQLearningAgent:
    """
    Double Q-Learning agent to reduce overestimation bias.
    
    Uses two Q-tables and randomly selects which one to update
    and which one to use for action selection.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.99,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """Initialize Double Q-Learning agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Two Q-tables
        self.q_table_1 = np.zeros((state_size, action_size))
        self.q_table_2 = np.zeros((state_size, action_size))
        
        # Performance tracking
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'epsilon_values': [],
            'success_rate': []
        }
    
    def act(self, state: int, training: bool = True) -> int:
        """Choose action using combined Q-values."""
        if training and random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            combined_q = self.q_table_1[state] + self.q_table_2[state]
            return np.argmax(combined_q)
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """Update Q-tables using Double Q-Learning."""
        if random.random() < 0.5:
            # Update Q1, use Q2 for target
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q_table_1[next_state])
                target = reward + self.discount_factor * self.q_table_2[next_state, best_action]
            
            self.q_table_1[state, action] += self.learning_rate * (target - self.q_table_1[state, action])
        else:
            # Update Q2, use Q1 for target
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q_table_2[next_state])
                target = reward + self.discount_factor * self.q_table_1[next_state, best_action]
            
            self.q_table_2[state, action] += self.learning_rate * (target - self.q_table_2[state, action])
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, episodes: int, verbose: bool = True) -> Dict[str, Any]:
        """Train the agent using Double Q-Learning."""
        start_time = time.time()
        success_count = 0
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                action = self.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            if done and info.get('position') == env.goal_position:
                success_count += 1
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(steps)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['success_rate'].append(success_count / (episode + 1))
            
            self.decay_epsilon()
            
            if verbose and (episode + 1) % 100 == 0:
                success_rate = success_count / (episode + 1)
                avg_reward = np.mean(self.training_history['rewards'][-100:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Success Rate: {success_rate:.3f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        return {
            'total_episodes': episodes,
            'success_rate': success_count / episodes,
            'avg_reward': np.mean(self.training_history['rewards']),
            'avg_steps': np.mean(self.training_history['steps']),
            'training_time': training_time,
            'final_epsilon': self.epsilon
        }
    
    def evaluate(self, env, episodes: int = 100) -> Dict[str, float]:
        """Evaluate the trained agent."""
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.training_history['rewards'])
        axes[0, 0].set_title('Rewards per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        axes[0, 1].plot(self.training_history['success_rate'])
        axes[0, 1].set_title('Success Rate Over Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        
        axes[1, 0].plot(self.training_history['steps'])
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        axes[1, 1].plot(self.training_history['epsilon_values'])
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.show()

