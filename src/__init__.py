"""
Grid World Navigation System

A comprehensive reinforcement learning system for grid world pathfinding.
Implements multiple algorithms including Q-Learning, SARSA, Double Q-Learning, and Deep Q-Networks.

Author: Mitash Shah
Date: May 2024 - Aug 2024
"""

__version__ = "1.0.0"
__author__ = "Mitash Shah"
__email__ = "mitashshah@example.com"

# Import main components
from .environment.grid_world import GridWorldEnv
from .algorithms.q_learning import QLearningAgent, SARSAAgent, DoubleQLearningAgent
from .algorithms.deep_q_learning import DQNAgent, DuelingDQNAgent, PrioritizedReplayDQNAgent
from .utils.parameter_optimizer import ParameterOptimizer
from .utils.benchmark import PerformanceBenchmark

__all__ = [
    "GridWorldEnv",
    "QLearningAgent",
    "SARSAAgent", 
    "DoubleQLearningAgent",
    "DQNAgent",
    "DuelingDQNAgent",
    "PrioritizedReplayDQNAgent",
    "ParameterOptimizer",
    "PerformanceBenchmark"
]
