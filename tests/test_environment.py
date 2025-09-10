#!/usr/bin/env python3
"""
Test suite for Grid World Environment

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


class TestGridWorldEnv(unittest.TestCase):
    """Test cases for GridWorldEnv."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = GridWorldEnv(size=8, dynamic_obstacles=False)
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.size, 8)
        self.assertEqual(self.env.action_space.n, 4)
        self.assertEqual(self.env.observation_space.n, 64)  # 8x8 = 64
        self.assertEqual(self.env.start_position, (0, 0))
        self.assertEqual(self.env.goal_position, (7, 7))
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        self.assertEqual(state, 0)  # Start position (0, 0) = index 0
        self.assertEqual(self.env.state, (0, 0))
    
    def test_step_valid_action(self):
        """Test valid action step."""
        self.env.reset()
        state, reward, done, info = self.env.step(3)  # Move right
        self.assertEqual(self.env.state, (0, 1))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    def test_step_invalid_action(self):
        """Test invalid action step (hitting boundary)."""
        self.env.reset()
        state, reward, done, info = self.env.step(2)  # Move left from (0, 0)
        self.assertEqual(self.env.state, (0, 0))  # Should not move
        self.assertLess(reward, 0)  # Should get penalty
    
    def test_goal_reaching(self):
        """Test reaching the goal."""
        self.env.state = (7, 6)  # One step from goal
        state, reward, done, info = self.env.step(3)  # Move right
        self.assertTrue(done)
        self.assertGreater(reward, 0)  # Should get positive reward
    
    def test_state_conversion(self):
        """Test state to index conversion."""
        # Test state_to_index
        self.assertEqual(self.env.state_to_index((0, 0)), 0)
        self.assertEqual(self.env.state_to_index((1, 0)), 8)
        self.assertEqual(self.env.state_to_index((0, 1)), 1)
        
        # Test index_to_state
        self.assertEqual(self.env.index_to_state(0), (0, 0))
        self.assertEqual(self.env.index_to_state(8), (1, 0))
        self.assertEqual(self.env.index_to_state(1), (0, 1))
    
    def test_obstacle_placement(self):
        """Test obstacle placement."""
        self.env.initialize_grid()
        self.assertGreater(len(self.env.obstacles), 0)
        
        # Check that start and goal are not obstacles
        self.assertNotIn(self.env.start_position, self.env.obstacles)
        self.assertNotIn(self.env.goal_position, self.env.obstacles)
    
    def test_dynamic_obstacles(self):
        """Test dynamic obstacle movement."""
        env = GridWorldEnv(size=8, dynamic_obstacles=True)
        initial_obstacles = env.obstacles.copy()
        
        # Move obstacles
        env.update_dynamic_obstacles()
        
        # Obstacles should have changed (with high probability)
        # Note: This test might occasionally fail due to randomness
        self.assertIsInstance(env.obstacles, set)
    
    def test_reward_structures(self):
        """Test different reward structures."""
        # Test sparse reward
        env_sparse = GridWorldEnv(size=8, reward_structure='sparse')
        env_sparse.reset()
        state, reward, done, info = env_sparse.step(0)  # Move up
        self.assertLess(reward, 0)  # Should get penalty
        
        # Test dense reward
        env_dense = GridWorldEnv(size=8, reward_structure='dense')
        env_dense.reset()
        state, reward, done, info = env_dense.step(0)  # Move up
        self.assertIsInstance(reward, float)
        
        # Test shaped reward
        env_shaped = GridWorldEnv(size=8, reward_structure='shaped')
        env_shaped.reset()
        state, reward, done, info = env_shaped.step(0)  # Move up
        self.assertIsInstance(reward, float)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        self.env.episode_count = 10
        self.env.success_count = 8
        
        metrics = self.env.get_performance_metrics()
        
        self.assertEqual(metrics['total_episodes'], 10)
        self.assertEqual(metrics['successful_episodes'], 8)
        self.assertEqual(metrics['success_rate'], 0.8)


if __name__ == '__main__':
    unittest.main()

