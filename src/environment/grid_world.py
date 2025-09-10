"""
Grid World Navigation Environment

A 16x16 grid world simulation for reinforcement learning pathfinding algorithms.
Supports dynamic obstacles and multiple reward structures for comprehensive testing.

Author: Mitash Shah
Date: May 2024 - Aug 2024
"""

import gym
from gym import spaces
import numpy as np
import random
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt


class GridWorldEnv(gym.Env):
    """
    Custom 16x16 Grid World Environment for reinforcement learning.
    
    Features:
    - 16x16 grid with configurable obstacles
    - Dynamic obstacle movement capability
    - Multiple reward structures
    - Comprehensive state representation
    - Performance tracking and metrics
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, size: int = 16, dynamic_obstacles: bool = False, 
                 reward_structure: str = 'sparse', obstacle_density: float = 0.1):
        """
        Initialize the Grid World Environment.
        
        Args:
            size: Grid size (default 16x16)
            dynamic_obstacles: Whether obstacles move during episodes
            reward_structure: Type of reward ('sparse', 'dense', 'shaped')
            obstacle_density: Fraction of grid cells that are obstacles
        """
        super(GridWorldEnv, self).__init__()
        
        self.size = size
        self.dynamic_obstacles = dynamic_obstacles
        self.reward_structure = reward_structure
        self.obstacle_density = obstacle_density
        
        # Action space: 4 directions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        # Observation space: flattened grid position
        self.observation_space = spaces.Discrete(self.size * self.size)
        
        # Environment state
        self.goal_position = (self.size - 1, self.size - 1)
        self.start_position = (0, 0)
        self.state = self.start_position
        self.obstacles = set()
        self.grid = None
        
        # Performance tracking
        self.episode_steps = 0
        self.total_reward = 0
        self.episode_count = 0
        self.success_count = 0
        
        # Initialize the environment
        self.reset()
    
    def initialize_grid(self) -> np.ndarray:
        """Initialize the grid with obstacles."""
        grid = np.zeros((self.size, self.size), dtype=int)
        self.obstacles = set()
        
        # Add static obstacles based on density
        num_obstacles = int(self.size * self.size * self.obstacle_density)
        obstacle_positions = set()
        
        # Ensure start and goal positions are not obstacles
        while len(obstacle_positions) < num_obstacles:
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if pos not in [self.start_position, self.goal_position]:
                obstacle_positions.add(pos)
        
        # Place obstacles
        for pos in obstacle_positions:
            grid[pos] = -1
            self.obstacles.add(pos)
        
        return grid
    
    def update_dynamic_obstacles(self) -> None:
        """Update obstacle positions for dynamic environment."""
        if not self.dynamic_obstacles or not self.obstacles:
            return
        
        new_obstacles = set()
        for (x, y) in self.obstacles:
            # Clear current obstacle
            self.grid[x, y] = 0
            
            # Random movement direction
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)])
            new_x, new_y = x + dx, y + dy
            
            # Check bounds and avoid goal/start
            if (0 <= new_x < self.size and 0 <= new_y < self.size and 
                (new_x, new_y) not in [self.goal_position, self.start_position]):
                new_obstacles.add((new_x, new_y))
            else:
                new_obstacles.add((x, y))
        
        self.obstacles = new_obstacles
        for (x, y) in self.obstacles:
            self.grid[x, y] = -1
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            next_state: Next state index
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # Action mapping
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dx, dy = moves[action]
        next_position = (self.state[0] + dx, self.state[1] + dy)
        
        # Check if move is valid
        valid_move = (0 <= next_position[0] < self.size and 
                     0 <= next_position[1] < self.size and 
                     next_position not in self.obstacles)
        
        if valid_move:
            self.state = next_position
        
        # Calculate reward based on structure
        reward = self._calculate_reward(valid_move)
        self.total_reward += reward
        self.episode_steps += 1
        
        # Check if goal reached
        done = (self.state == self.goal_position)
        if done:
            self.success_count += 1
        
        # Update dynamic obstacles
        self.update_dynamic_obstacles()
        
        # Prepare info
        info = {
            'valid_move': valid_move,
            'steps': self.episode_steps,
            'total_reward': self.total_reward,
            'position': self.state
        }
        
        return self.state_to_index(self.state), reward, done, info
    
    def _calculate_reward(self, valid_move: bool) -> float:
        """Calculate reward based on current reward structure."""
        if not valid_move:
            return -0.1  # Small penalty for invalid moves
        
        if self.reward_structure == 'sparse':
            # Sparse reward: only reward at goal
            return 100.0 if self.state == self.goal_position else -0.1
        
        elif self.reward_structure == 'dense':
            # Dense reward: distance-based
            if self.state == self.goal_position:
                return 100.0
            else:
                # Manhattan distance to goal
                distance = abs(self.state[0] - self.goal_position[0]) + \
                          abs(self.state[1] - self.goal_position[1])
                return -0.1 - (distance / (2 * self.size))
        
        elif self.reward_structure == 'shaped':
            # Shaped reward: encourages progress toward goal
            if self.state == self.goal_position:
                return 100.0
            else:
                # Reward based on progress toward goal
                current_distance = abs(self.state[0] - self.goal_position[0]) + \
                                 abs(self.state[1] - self.goal_position[1])
                max_distance = 2 * (self.size - 1)
                progress_reward = (max_distance - current_distance) / max_distance
                return -0.1 + 0.5 * progress_reward
        
        return -0.1
    
    def reset(self) -> int:
        """Reset the environment to initial state."""
        self.state = self.start_position
        self.grid = self.initialize_grid()
        self.episode_steps = 0
        self.total_reward = 0
        self.episode_count += 1
        
        return self.state_to_index(self.state)
    
    def render(self, mode: str = 'human') -> None:
        """Render the current state of the environment."""
        if mode == 'human':
            grid_display = np.array(self.grid, dtype=str)
            grid_display[grid_display == '0'] = ' '  # Free space
            grid_display[grid_display == '-1'] = 'X'  # Obstacles
            grid_display[self.goal_position[0]][self.goal_position[1]] = 'G'  # Goal
            grid_display[self.state[0]][self.state[1]] = 'A'  # Agent
            
            print(f"\nEpisode {self.episode_count}, Step {self.episode_steps}")
            print("=" * (self.size * 2 + 1))
            for row in grid_display:
                print("|" + " ".join(row) + "|")
            print("=" * (self.size * 2 + 1))
            print(f"Position: {self.state}, Reward: {self.total_reward:.2f}")
            print()
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (x, y) state to single index."""
        return state[0] * self.size + state[1]
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert single index to (x, y) state."""
        return divmod(index, self.size)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        success_rate = self.success_count / max(self.episode_count, 1)
        avg_steps = self.episode_steps / max(self.episode_count, 1)
        
        return {
            'success_rate': success_rate,
            'avg_steps_per_episode': avg_steps,
            'total_episodes': self.episode_count,
            'successful_episodes': self.success_count
        }
    
    def visualize_grid(self, title: str = "Grid World") -> None:
        """Visualize the current grid state."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create visualization grid
        vis_grid = np.zeros((self.size, self.size))
        vis_grid[self.state] = 0.5  # Agent
        vis_grid[self.goal_position] = 1.0  # Goal
        
        # Mark obstacles
        for (x, y) in self.obstacles:
            vis_grid[x, y] = -0.5
        
        # Plot
        im = ax.imshow(vis_grid, cmap='RdYlBu', vmin=-0.5, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel('Y Coordinate')
        ax.set_ylabel('X Coordinate')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cell Type')
        
        plt.tight_layout()
        plt.show()

