#!/usr/bin/env python3
"""
Grid World Navigation System - Main Execution Script

Comprehensive reinforcement learning system for grid world pathfinding.
Implements Q-Learning, SARSA, Double Q-Learning, and Deep Q-Learning algorithms.

Author: Mitash Shah
Date: May 2024 - Aug 2024
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment.grid_world import GridWorldEnv
from algorithms.q_learning import QLearningAgent, SARSAAgent, DoubleQLearningAgent
from algorithms.deep_q_learning import DQNAgent, DuelingDQNAgent, PrioritizedReplayDQNAgent
from utils.parameter_optimizer import ParameterOptimizer
from utils.benchmark import PerformanceBenchmark


def run_algorithm_comparison():
    """Run comprehensive algorithm comparison."""
    print("=" * 80)
    print("GRID WORLD NAVIGATION - ALGORITHM COMPARISON")
    print("=" * 80)
    
    # Create environment
    env = GridWorldEnv(size=16, dynamic_obstacles=True, reward_structure='shaped')
    
    # Initialize algorithms
    algorithms = {
        'Q-Learning': QLearningAgent(env.observation_space.n, env.action_space.n),
        'SARSA': SARSAAgent(env.observation_space.n, env.action_space.n),
        'Double Q-Learning': DoubleQLearningAgent(env.observation_space.n, env.action_space.n),
        'Deep Q-Network': DQNAgent(env.observation_space.n, env.action_space.n),
        'Dueling DQN': DuelingDQNAgent(env.observation_space.n, env.action_space.n),
        'Prioritized Replay DQN': PrioritizedReplayDQNAgent(env.observation_space.n, env.action_space.n)
    }
    
    # Benchmark system
    benchmark = PerformanceBenchmark(evaluation_episodes=100, n_trials=3)
    
    # Evaluate each algorithm
    results = {}
    for name, agent in algorithms.items():
        print(f"\nEvaluating {name}...")
        results[name] = benchmark.evaluate_algorithm(name, agent, env, training_episodes=1000)
    
    # Compare algorithms
    comparison = benchmark.compare_algorithms(results)
    
    # Display results
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 80)
    
    for i, ranking in enumerate(comparison['ranking']):
        print(f"{i+1}. {ranking['algorithm']}")
        print(f"   Success Rate: {ranking['success_rate']:.3f}")
        print(f"   Pathfinding Time: {ranking['pathfinding_time']:.1f} steps")
        print(f"   Improvement: {ranking['improvement_percentage']:.1f}%")
        print(f"   Composite Score: {ranking['composite_score']:.3f}")
        print()
    
    # Generate performance report
    report = benchmark.generate_performance_report()
    print(report)
    
    # Plot results
    benchmark.plot_performance_comparison('results/plots/algorithm_comparison.png')
    
    # Save results
    benchmark.save_benchmark_results('results/benchmark_results.json')
    
    return comparison


def run_parameter_optimization():
    """Run parameter optimization for best algorithm."""
    print("=" * 80)
    print("PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Optimize Q-Learning (typically best for this problem)
    optimizer = ParameterOptimizer(QLearningAgent, GridWorldEnv, evaluation_episodes=50, n_trials=2)
    
    # Run random search optimization
    optimization_results = optimizer.random_search('q_learning', n_samples=20)
    
    print(f"\nBest Parameters: {optimization_results['best_parameters']}")
    print(f"Best Score: {optimization_results['best_score']:.3f}")
    
    # Plot optimization results
    optimizer.plot_optimization_results('results/plots/parameter_optimization.png')
    
    # Save results
    optimizer.save_results('results/parameter_optimization.json')
    
    return optimization_results


def run_performance_demonstration():
    """Run performance demonstration with best parameters."""
    print("=" * 80)
    print("PERFORMANCE DEMONSTRATION")
    print("=" * 80)
    
    # Create environment
    env = GridWorldEnv(size=16, dynamic_obstacles=True, reward_structure='shaped')
    
    # Use optimized parameters
    best_params = {
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01
    }
    
    # Create and train agent
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, **best_params)
    
    print("Training agent with optimized parameters...")
    training_stats = agent.train(env, episodes=2000, verbose=True)
    
    # Evaluate performance
    print("\nEvaluating performance...")
    eval_results = agent.evaluate(env, episodes=100)
    
    print(f"\nPerformance Results:")
    print(f"Success Rate: {eval_results['success_rate']:.3f}")
    print(f"Average Steps: {eval_results['avg_steps']:.1f}")
    print(f"Improvement: {((50 - eval_results['avg_steps']) / 50) * 100:.1f}%")
    
    # Visualize training progress
    agent.plot_training_progress()
    
    # Demonstrate pathfinding
    print("\nDemonstrating pathfinding...")
    env.reset()
    env.render()
    
    state = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 100:
        action = agent.act(state, training=False)
        state, reward, done, info = env.step(action)
        steps += 1
        
        if steps % 10 == 0:
            env.render()
    
    if done:
        print(f"Goal reached in {steps} steps!")
    else:
        print("Failed to reach goal within 100 steps.")
    
    return eval_results


def run_algorithm_showcase():
    """Showcase different algorithms with visualizations."""
    print("=" * 80)
    print("ALGORITHM SHOWCASE")
    print("=" * 80)
    
    # Create environment
    env = GridWorldEnv(size=16, dynamic_obstacles=False, reward_structure='sparse')
    
    # Algorithms to showcase
    algorithms = {
        'Q-Learning': QLearningAgent(env.observation_space.n, env.action_space.n),
        'SARSA': SARSAAgent(env.observation_space.n, env.action_space.n),
        'Double Q-Learning': DoubleQLearningAgent(env.observation_space.n, env.action_space.n)
    }
    
    # Train and visualize each algorithm
    for name, agent in algorithms.items():
        print(f"\nTraining {name}...")
        agent.train(env, episodes=1000, verbose=False)
        
        # Evaluate
        eval_results = agent.evaluate(env, episodes=50)
        print(f"Success Rate: {eval_results['success_rate']:.3f}")
        print(f"Average Steps: {eval_results['avg_steps']:.1f}")
        
        # Plot training progress
        agent.plot_training_progress()
    
    return algorithms


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Grid World Navigation System')
    parser.add_argument('--mode', choices=['compare', 'optimize', 'demo', 'showcase', 'all'],
                       default='all', help='Execution mode')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--size', type=int, default=16,
                       help='Grid size')
    parser.add_argument('--dynamic', action='store_true',
                       help='Use dynamic obstacles')
    
    args = parser.parse_args()
    
    # Create results directory
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'compare' or args.mode == 'all':
        run_algorithm_comparison()
    
    if args.mode == 'optimize' or args.mode == 'all':
        run_parameter_optimization()
    
    if args.mode == 'demo' or args.mode == 'all':
        run_performance_demonstration()
    
    if args.mode == 'showcase' or args.mode == 'all':
        run_algorithm_showcase()
    
    print("\n" + "=" * 80)
    print("GRID WORLD NAVIGATION SYSTEM - EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

