"""
Parameter Optimization Module

Comprehensive hyperparameter tuning for reinforcement learning algorithms.
Implements grid search, random search, and Bayesian optimization.

Author: Mitash Shah
Date: May 2024 - Aug 2024
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple, Any, Callable, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
import json
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


class ParameterOptimizer:
    """
    Comprehensive parameter optimization for RL algorithms.
    
    Features:
    - Grid search optimization
    - Random search optimization
    - Performance comparison and visualization
    - Parameter sensitivity analysis
    - Automated best parameter selection
    """
    
    def __init__(self, algorithm_class, environment_class, 
                 evaluation_episodes: int = 100, n_trials: int = 3):
        """
        Initialize parameter optimizer.
        
        Args:
            algorithm_class: RL algorithm class to optimize
            environment_class: Environment class to use
            evaluation_episodes: Number of episodes for evaluation
            n_trials: Number of trials per parameter combination
        """
        self.algorithm_class = algorithm_class
        self.environment_class = environment_class
        self.evaluation_episodes = evaluation_episodes
        self.n_trials = n_trials
        
        # Results storage
        self.optimization_results = []
        self.best_parameters = None
        self.best_score = -np.inf
        
        # Parameter spaces for different algorithms
        self.parameter_spaces = self._define_parameter_spaces()
    
    def _define_parameter_spaces(self) -> Dict[str, Dict]:
        """Define parameter search spaces for different algorithms."""
        return {
            'q_learning': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'discount_factor': [0.9, 0.95, 0.99, 0.995],
                'epsilon': [0.1, 0.2, 0.3, 0.5],
                'epsilon_decay': [0.99, 0.995, 0.999],
                'epsilon_min': [0.01, 0.05, 0.1]
            },
            'sarsa': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'discount_factor': [0.9, 0.95, 0.99, 0.995],
                'epsilon': [0.1, 0.2, 0.3, 0.5],
                'epsilon_decay': [0.99, 0.995, 0.999],
                'epsilon_min': [0.01, 0.05, 0.1]
            },
            'double_q_learning': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'discount_factor': [0.9, 0.95, 0.99, 0.995],
                'epsilon': [0.1, 0.2, 0.3, 0.5],
                'epsilon_decay': [0.99, 0.995, 0.999],
                'epsilon_min': [0.01, 0.05, 0.1]
            },
            'dqn': {
                'learning_rate': [0.0001, 0.001, 0.01],
                'discount_factor': [0.9, 0.95, 0.99],
                'epsilon': [0.1, 0.2, 0.3],
                'epsilon_decay': [0.99, 0.995, 0.999],
                'epsilon_min': [0.01, 0.05],
                'batch_size': [16, 32, 64],
                'memory_size': [1000, 2000, 5000],
                'target_update_freq': [5, 10, 20]
            }
        }
    
    def _evaluate_parameters(self, params: Dict[str, Any], 
                           algorithm_name: str) -> Dict[str, float]:
        """
        Evaluate a parameter combination.
        
        Args:
            params: Parameter dictionary
            algorithm_name: Name of the algorithm
            
        Returns:
            Evaluation metrics
        """
        scores = []
        training_times = []
        
        for trial in range(self.n_trials):
            # Create environment
            env = self.environment_class(size=16, dynamic_obstacles=True)
            
            # Create agent with parameters
            if algorithm_name in ['q_learning', 'sarsa', 'double_q_learning']:
                agent = self.algorithm_class(
                    state_size=env.observation_space.n,
                    action_size=env.action_space.n,
                    **params
                )
            else:  # DQN variants
                agent = self.algorithm_class(
                    state_size=env.observation_space.n,
                    action_size=env.action_space.n,
                    **params
                )
            
            # Train agent
            start_time = time.time()
            training_stats = agent.train(env, episodes=1000, verbose=False)
            training_time = time.time() - start_time
            
            # Evaluate agent
            eval_results = agent.evaluate(env, episodes=self.evaluation_episodes)
            
            scores.append(eval_results['success_rate'])
            training_times.append(training_time)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_training_time': np.mean(training_times),
            'scores': scores,
            'training_times': training_times
        }
    
    def grid_search(self, algorithm_name: str, 
                   custom_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform grid search optimization.
        
        Args:
            algorithm_name: Name of algorithm to optimize
            custom_params: Custom parameter space (optional)
            
        Returns:
            Optimization results
        """
        print(f"Starting grid search optimization for {algorithm_name}...")
        
        # Get parameter space
        if custom_params:
            param_space = custom_params
        else:
            param_space = self.parameter_spaces.get(algorithm_name, {})
        
        if not param_space:
            raise ValueError(f"No parameter space defined for {algorithm_name}")
        
        # Generate parameter combinations
        param_grid = ParameterGrid(param_space)
        total_combinations = len(param_grid)
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        results = []
        start_time = time.time()
        
        for i, params in enumerate(param_grid):
            print(f"Testing combination {i+1}/{total_combinations}: {params}")
            
            try:
                eval_results = self._evaluate_parameters(params, algorithm_name)
                
                result = {
                    'parameters': params,
                    'mean_score': eval_results['mean_score'],
                    'std_score': eval_results['std_score'],
                    'mean_training_time': eval_results['mean_training_time'],
                    'scores': eval_results['scores'],
                    'training_times': eval_results['training_times']
                }
                
                results.append(result)
                
                # Update best parameters
                if eval_results['mean_score'] > self.best_score:
                    self.best_score = eval_results['mean_score']
                    self.best_parameters = params.copy()
                
                print(f"  Score: {eval_results['mean_score']:.3f} ± {eval_results['std_score']:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        optimization_time = time.time() - start_time
        
        # Store results
        self.optimization_results = results
        
        return {
            'algorithm': algorithm_name,
            'method': 'grid_search',
            'total_combinations': total_combinations,
            'successful_combinations': len(results),
            'best_score': self.best_score,
            'best_parameters': self.best_parameters,
            'optimization_time': optimization_time,
            'results': results
        }
    
    def random_search(self, algorithm_name: str, n_samples: int = 50,
                     custom_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform random search optimization.
        
        Args:
            algorithm_name: Name of algorithm to optimize
            n_samples: Number of random samples
            custom_params: Custom parameter space (optional)
            
        Returns:
            Optimization results
        """
        print(f"Starting random search optimization for {algorithm_name}...")
        
        # Get parameter space
        if custom_params:
            param_space = custom_params
        else:
            param_space = self.parameter_spaces.get(algorithm_name, {})
        
        if not param_space:
            raise ValueError(f"No parameter space defined for {algorithm_name}")
        
        results = []
        start_time = time.time()
        
        for i in range(n_samples):
            # Sample random parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    params[param_name] = np.random.choice(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # Continuous range
                    params[param_name] = np.random.uniform(param_values[0], param_values[1])
            
            print(f"Testing sample {i+1}/{n_samples}: {params}")
            
            try:
                eval_results = self._evaluate_parameters(params, algorithm_name)
                
                result = {
                    'parameters': params,
                    'mean_score': eval_results['mean_score'],
                    'std_score': eval_results['std_score'],
                    'mean_training_time': eval_results['mean_training_time'],
                    'scores': eval_results['scores'],
                    'training_times': eval_results['training_times']
                }
                
                results.append(result)
                
                # Update best parameters
                if eval_results['mean_score'] > self.best_score:
                    self.best_score = eval_results['mean_score']
                    self.best_parameters = params.copy()
                
                print(f"  Score: {eval_results['mean_score']:.3f} ± {eval_results['std_score']:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        optimization_time = time.time() - start_time
        
        # Store results
        self.optimization_results = results
        
        return {
            'algorithm': algorithm_name,
            'method': 'random_search',
            'n_samples': n_samples,
            'successful_samples': len(results),
            'best_score': self.best_score,
            'best_parameters': self.best_parameters,
            'optimization_time': optimization_time,
            'results': results
        }
    
    def compare_algorithms(self, algorithms: List[str], 
                          n_samples: int = 20) -> Dict[str, Any]:
        """
        Compare multiple algorithms using random search.
        
        Args:
            algorithms: List of algorithm names to compare
            n_samples: Number of samples per algorithm
            
        Returns:
            Comparison results
        """
        print(f"Comparing algorithms: {algorithms}")
        
        comparison_results = {}
        
        for algorithm in algorithms:
            print(f"\nOptimizing {algorithm}...")
            results = self.random_search(algorithm, n_samples=n_samples)
            comparison_results[algorithm] = results
        
        # Find best overall algorithm
        best_algorithm = max(comparison_results.keys(), 
                           key=lambda x: comparison_results[x]['best_score'])
        
        return {
            'algorithms': algorithms,
            'comparison_results': comparison_results,
            'best_algorithm': best_algorithm,
            'best_overall_score': comparison_results[best_algorithm]['best_score']
        }
    
    def plot_optimization_results(self, save_path: Optional[str] = None) -> None:
        """Plot optimization results."""
        if not self.optimization_results:
            print("No optimization results to plot.")
            return
        
        # Extract data for plotting
        scores = [r['mean_score'] for r in self.optimization_results]
        std_scores = [r['std_score'] for r in self.optimization_results]
        training_times = [r['mean_training_time'] for r in self.optimization_results]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Score distribution
        axes[0, 0].hist(scores, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.best_score, color='red', linestyle='--', 
                          label=f'Best Score: {self.best_score:.3f}')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].set_xlabel('Success Rate')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Score vs Training Time
        scatter = axes[0, 1].scatter(training_times, scores, 
                                   c=std_scores, cmap='viridis', alpha=0.7)
        axes[0, 1].set_title('Score vs Training Time')
        axes[0, 1].set_xlabel('Training Time (seconds)')
        axes[0, 1].set_ylabel('Success Rate')
        plt.colorbar(scatter, ax=axes[0, 1], label='Score Std Dev')
        
        # Top 10 results
        top_10 = sorted(self.optimization_results, 
                       key=lambda x: x['mean_score'], reverse=True)[:10]
        top_scores = [r['mean_score'] for r in top_10]
        top_indices = list(range(len(top_scores)))
        
        axes[1, 0].bar(top_indices, top_scores, alpha=0.7)
        axes[1, 0].set_title('Top 10 Parameter Combinations')
        axes[1, 0].set_xlabel('Rank')
        axes[1, 0].set_ylabel('Success Rate')
        
        # Score progression
        axes[1, 1].plot(scores, alpha=0.7)
        axes[1, 1].set_title('Score Progression')
        axes[1, 1].set_xlabel('Parameter Combination')
        axes[1, 1].set_ylabel('Success Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_parameter_sensitivity(self, param_name: str) -> Dict[str, float]:
        """
        Analyze sensitivity of a specific parameter.
        
        Args:
            param_name: Name of parameter to analyze
            
        Returns:
            Sensitivity analysis results
        """
        if not self.optimization_results:
            return {}
        
        # Group results by parameter value
        param_values = []
        scores = []
        
        for result in self.optimization_results:
            if param_name in result['parameters']:
                param_values.append(result['parameters'][param_name])
                scores.append(result['mean_score'])
        
        if not param_values:
            return {}
        
        # Calculate statistics
        unique_values = sorted(set(param_values))
        value_scores = {}
        
        for value in unique_values:
            value_scores[value] = [scores[i] for i, v in enumerate(param_values) if v == value]
        
        sensitivity = {}
        for value, value_score_list in value_scores.items():
            sensitivity[value] = {
                'mean': np.mean(value_score_list),
                'std': np.std(value_score_list),
                'count': len(value_score_list)
            }
        
        return sensitivity
    
    def save_results(self, filepath: str) -> None:
        """Save optimization results to file."""
        results_data = {
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'optimization_results': self.optimization_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def load_results(self, filepath: str) -> None:
        """Load optimization results from file."""
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.best_parameters = results_data['best_parameters']
        self.best_score = results_data['best_score']
        self.optimization_results = results_data['optimization_results']

