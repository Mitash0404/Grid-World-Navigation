#!/usr/bin/env python3
"""
Test suite for RL Algorithms

Author: Mitash Shah
Date: May 2024 - Aug 2024
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.grid_world import GridWorldEnv
from algorithms.q_learning import QLearningAgent, SARSAAgent, DoubleQLearningAgent


class TestQLearningAgent(unittest.TestCase):
    """Test cases for QLearningAgent."""
    
    def setUp(self):
        """Set up test environment and agent."""
        self.env = GridWorldEnv(size=4, dynamic_obstacles=False)
        self.agent = QLearningAgent(
            state_size=self.env.observation_space.n,
            action_size=self.env.action_space.n,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=0.1
        )
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_size, 16)  # 4x4 = 16
        self.assertEqual(self.agent.action_size, 4)
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.99)
        self.assertEqual(self.agent.epsilon, 0.1)
    
    def test_q_table_initialization(self):
        """Test Q-table initialization."""
        self.assertEqual(self.agent.q_table.shape, (16, 4))
        self.assertTrue(np.all(self.agent.q_table == 0))
    
    def test_act_exploration(self):
        """Test action selection during exploration."""
        # Set high epsilon to force exploration
        self.agent.epsilon = 1.0
        actions = [self.agent.act(0, training=True) for _ in range(100)]
        
        # Should have some variety in actions
        unique_actions = len(set(actions))
        self.assertGreater(unique_actions, 1)
    
    def test_act_exploitation(self):
        """Test action selection during exploitation."""
        # Set low epsilon to force exploitation
        self.agent.epsilon = 0.0
        
        # Set up Q-table with clear best action
        self.agent.q_table[0, 1] = 10.0  # Action 1 is best for state 0
        
        action = self.agent.act(0, training=False)
        self.assertEqual(action, 1)
    
    def test_update(self):
        """Test Q-table update."""
        initial_q = self.agent.q_table[0, 1].copy()
        
        self.agent.update(0, 1, 10.0, 1, False)
        
        # Q-value should have changed
        self.assertNotEqual(self.agent.q_table[0, 1], initial_q)
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()
        
        self.assertLess(self.agent.epsilon, initial_epsilon)
    
    def test_training(self):
        """Test agent training."""
        training_stats = self.agent.train(self.env, episodes=10, verbose=False)
        
        self.assertIn('total_episodes', training_stats)
        self.assertIn('success_rate', training_stats)
        self.assertIn('avg_reward', training_stats)
        self.assertEqual(training_stats['total_episodes'], 10)
    
    def test_evaluation(self):
        """Test agent evaluation."""
        # Train agent briefly
        self.agent.train(self.env, episodes=5, verbose=False)
        
        # Evaluate
        eval_results = self.agent.evaluate(self.env, episodes=10)
        
        self.assertIn('success_rate', eval_results)
        self.assertIn('avg_reward', eval_results)
        self.assertIn('avg_steps', eval_results)
        self.assertGreaterEqual(eval_results['success_rate'], 0.0)
        self.assertLessEqual(eval_results['success_rate'], 1.0)


class TestSARSAAgent(unittest.TestCase):
    """Test cases for SARSAAgent."""
    
    def setUp(self):
        """Set up test environment and agent."""
        self.env = GridWorldEnv(size=4, dynamic_obstacles=False)
        self.agent = SARSAAgent(
            state_size=self.env.observation_space.n,
            action_size=self.env.action_space.n
        )
    
    def test_sarsa_update(self):
        """Test SARSA update rule."""
        initial_q = self.agent.q_table[0, 1].copy()
        
        # SARSA update with next action
        self.agent.update(0, 1, 10.0, 1, 2, False)
        
        # Q-value should have changed
        self.assertNotEqual(self.agent.q_table[0, 1], initial_q)
    
    def test_sarsa_training(self):
        """Test SARSA training."""
        training_stats = self.agent.train(self.env, episodes=10, verbose=False)
        
        self.assertEqual(training_stats['total_episodes'], 10)
        self.assertIn('success_rate', training_stats)


class TestDoubleQLearningAgent(unittest.TestCase):
    """Test cases for DoubleQLearningAgent."""
    
    def setUp(self):
        """Set up test environment and agent."""
        self.env = GridWorldEnv(size=4, dynamic_obstacles=False)
        self.agent = DoubleQLearningAgent(
            state_size=self.env.observation_space.n,
            action_size=self.env.action_space.n
        )
    
    def test_double_q_tables(self):
        """Test double Q-table initialization."""
        self.assertEqual(self.agent.q_table_1.shape, (16, 4))
        self.assertEqual(self.agent.q_table_2.shape, (16, 4))
        self.assertTrue(np.all(self.agent.q_table_1 == 0))
        self.assertTrue(np.all(self.agent.q_table_2 == 0))
    
    def test_double_q_update(self):
        """Test double Q-learning update."""
        initial_q1 = self.agent.q_table_1[0, 1].copy()
        initial_q2 = self.agent.q_table_2[0, 1].copy()
        
        self.agent.update(0, 1, 10.0, 1, False)
        
        # At least one Q-table should have changed
        q1_changed = self.agent.q_table_1[0, 1] != initial_q1
        q2_changed = self.agent.q_table_2[0, 1] != initial_q2
        self.assertTrue(q1_changed or q2_changed)
    
    def test_double_q_training(self):
        """Test double Q-learning training."""
        training_stats = self.agent.train(self.env, episodes=10, verbose=False)
        
        self.assertEqual(training_stats['total_episodes'], 10)
        self.assertIn('success_rate', training_stats)


if __name__ == '__main__':
    unittest.main()
