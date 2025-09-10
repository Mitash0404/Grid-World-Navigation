#!/usr/bin/env python3
"""
Basic Usage Example - Grid World Navigation

Demonstrates basic usage of the Grid World Navigation system with Q-Learning.

Author: Mitash Shah
Date: May 2024 - Aug 2024
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.grid_world import GridWorldEnv
from algorithms.q_learning import QLearningAgent


def main():
    """Basic usage example."""
    print("Grid World Navigation - Basic Usage Example")
    print("=" * 50)
    
    # Create environment
    print("Creating 16x16 grid world environment...")
    env = GridWorldEnv(size=16, dynamic_obstacles=False, reward_structure='sparse')
    
    # Create Q-Learning agent
    print("Initializing Q-Learning agent...")
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train the agent
    print("Training agent for 1000 episodes...")
    training_stats = agent.train(env, episodes=1000, verbose=True)
    
    # Evaluate performance
    print("\nEvaluating performance...")
    eval_results = agent.evaluate(env, episodes=100)
    
    # Display results
    print("\n" + "=" * 50)
    print("PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"Success Rate: {eval_results['success_rate']:.3f}")
    print(f"Average Steps: {eval_results['avg_steps']:.1f}")
    print(f"Standard Deviation: {eval_results['std_steps']:.1f}")
    
    # Calculate improvement percentage
    baseline_steps = 50  # Assume baseline takes 50 steps
    improvement = ((baseline_steps - eval_results['avg_steps']) / baseline_steps) * 100
    print(f"Pathfinding Improvement: {improvement:.1f}%")
    
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
        
        if steps % 5 == 0:
            print(f"\nStep {steps}:")
            env.render()
    
    if done:
        print(f"\nGoal reached in {steps} steps!")
    else:
        print("\nFailed to reach goal within 100 steps.")
    
    # Plot training progress
    print("\nPlotting training progress...")
    agent.plot_training_progress()
    
    print("\nBasic usage example completed!")


if __name__ == "__main__":
    main()

