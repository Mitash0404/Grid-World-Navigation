#!/usr/bin/env python3
"""
Algorithm Comparison Example

Compares different reinforcement learning algorithms on the grid world task.

Author: Mitash Shah
Date: May 2024 - Aug 2024
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.grid_world import GridWorldEnv
from algorithms.q_learning import QLearningAgent, SARSAAgent, DoubleQLearningAgent
from algorithms.deep_q_learning import DQNAgent, DuelingDQNAgent
from utils.benchmark import PerformanceBenchmark


def main():
    """Algorithm comparison example."""
    print("Grid World Navigation - Algorithm Comparison")
    print("=" * 60)
    
    # Create environment
    print("Creating environment...")
    env = GridWorldEnv(size=16, dynamic_obstacles=True, reward_structure='shaped')
    
    # Initialize algorithms
    algorithms = {
        'Q-Learning': QLearningAgent(env.observation_space.n, env.action_space.n),
        'SARSA': SARSAAgent(env.observation_space.n, env.action_space.n),
        'Double Q-Learning': DoubleQLearningAgent(env.observation_space.n, env.action_space.n),
        'Deep Q-Network': DQNAgent(env.observation_space.n, env.action_space.n),
        'Dueling DQN': DuelingDQNAgent(env.observation_space.n, env.action_space.n)
    }
    
    # Benchmark system
    benchmark = PerformanceBenchmark(evaluation_episodes=50, n_trials=2)
    
    # Evaluate each algorithm
    print("\nEvaluating algorithms...")
    results = {}
    
    for name, agent in algorithms.items():
        print(f"\nEvaluating {name}...")
        results[name] = benchmark.evaluate_algorithm(name, agent, env, training_episodes=500)
    
    # Compare algorithms
    print("\nComparing algorithms...")
    comparison = benchmark.compare_algorithms(results)
    
    # Display results
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"{'Rank':<4} {'Algorithm':<20} {'Success Rate':<12} {'Avg Steps':<10} {'Improvement':<12}")
    print("-" * 60)
    
    for i, ranking in enumerate(comparison['ranking']):
        print(f"{i+1:<4} {ranking['algorithm']:<20} {ranking['success_rate']:<12.3f} "
              f"{ranking['pathfinding_time']:<10.1f} {ranking['improvement_percentage']:<12.1f}%")
    
    # Statistical analysis
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    success_comparison = comparison['success_rate_comparison']
    print(f"ANOVA F-statistic: {success_comparison['anova_f_statistic']:.3f}")
    print(f"ANOVA p-value: {success_comparison['anova_p_value']:.3f}")
    print(f"Significant difference: {success_comparison['anova_significant']}")
    
    # Performance summary
    summary = comparison['performance_summary']
    print(f"\nBest Success Rate: {summary['best_success_rate']:.3f} ({summary['best_success_rate_algorithm']})")
    print(f"Best Speed: {summary['best_pathfinding_time']:.1f} steps ({summary['best_speed_algorithm']})")
    print(f"Best Improvement: {summary['best_improvement']:.1f}% ({summary['best_improvement_algorithm']})")
    
    # Plot results
    print("\nGenerating performance plots...")
    benchmark.plot_performance_comparison('results/plots/algorithm_comparison.png')
    
    # Save results
    benchmark.save_benchmark_results('results/algorithm_comparison.json')
    
    print("\nAlgorithm comparison completed!")
    print("Results saved to results/ directory.")


if __name__ == "__main__":
    main()

