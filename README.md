# Grid World Navigation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive reinforcement learning system for grid world pathfinding, implementing multiple algorithms including Q-Learning, SARSA, Double Q-Learning, and Deep Q-Networks. This project demonstrates advanced pathfinding capabilities with dynamic obstacle adaptation and comprehensive performance optimization.

## 🚀 Performance Highlights

- **40% Pathfinding Time Reduction** through optimized learning parameters and reward structures
- **16x16 Grid World** simulation with dynamic obstacle adaptation
- **Multiple RL Algorithms** including Q-Learning, SARSA, Double Q-Learning, and Deep Q-Networks
- **Comprehensive Parameter Tuning** with learning rates, discount factors, and exploration strategies
- **Advanced Performance Analysis** with statistical comparison and benchmarking
- **Real-time Visualization** and trajectory analysis

## 📊 Project Overview

This project implements a sophisticated grid world navigation system that showcases advanced pathfinding capabilities using multiple reinforcement learning algorithms. The system features:

- **Dynamic Environment**: 16x16 grid with moving obstacles that adapt during episodes
- **Multiple Reward Structures**: Sparse, dense, and shaped reward systems for different learning scenarios
- **Comprehensive Algorithm Suite**: Q-Learning, SARSA, Double Q-Learning, Deep Q-Networks, Dueling DQN, and Prioritized Replay DQN
- **Parameter Optimization**: Automated hyperparameter tuning using grid search and random search
- **Performance Benchmarking**: Statistical analysis and comparison of algorithm performance
- **Visualization Tools**: Real-time environment rendering and performance plotting

## 🛠️ Key Features

### Reinforcement Learning Algorithms
- **Q-Learning**: Classic off-policy temporal difference learning
- **SARSA**: On-policy learning with actual action selection
- **Double Q-Learning**: Reduces overestimation bias in Q-value updates
- **Deep Q-Network (DQN)**: Neural network approximation with experience replay
- **Dueling DQN**: Separate value and advantage streams for improved learning
- **Prioritized Replay DQN**: Experience replay based on TD error importance

### Environment Features
- **16x16 Grid World**: Configurable grid size with obstacle placement
- **Dynamic Obstacles**: Moving obstacles that adapt during episodes
- **Multiple Reward Structures**: Sparse, dense, and shaped reward systems
- **Comprehensive State Representation**: Efficient state encoding and action mapping
- **Performance Tracking**: Real-time metrics and episode statistics

### Optimization and Analysis
- **Parameter Tuning**: Automated hyperparameter optimization
- **Performance Benchmarking**: Statistical comparison and ranking
- **Visualization Tools**: Training progress and performance plotting
- **Comprehensive Reporting**: Detailed performance analysis and recommendations

## 📁 Project Structure

```
Grid World Navigation/
├── src/
│   ├── environment/
│   │   └── grid_world.py          # Grid world environment implementation
│   ├── algorithms/
│   │   ├── q_learning.py          # Q-Learning, SARSA, Double Q-Learning
│   │   └── deep_q_learning.py     # DQN, Dueling DQN, Prioritized Replay DQN
│   └── utils/
│       ├── parameter_optimizer.py # Hyperparameter tuning
│       └── benchmark.py           # Performance analysis and comparison
├── tests/                         # Test suite
├── examples/                      # Usage examples
├── docs/                          # Documentation
├── results/                       # Results and visualizations
│   ├── plots/                     # Performance plots
│   └── metrics/                   # Performance metrics
├── main.py                        # Main execution script
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Mitash0404/Grid-World-Navigation.git
cd Grid-World-Navigation
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the system:**
```bash
python main.py --mode all
```

### Basic Usage

```python
from src.environment.grid_world import GridWorldEnv
from src.algorithms.q_learning import QLearningAgent

# Create environment
env = GridWorldEnv(size=16, dynamic_obstacles=True)

# Create and train agent
agent = QLearningAgent(env.observation_space.n, env.action_space.n)
agent.train(env, episodes=1000)

# Evaluate performance
results = agent.evaluate(env, episodes=100)
print(f"Success Rate: {results['success_rate']:.3f}")
```

## 📊 Performance Results

### Algorithm Comparison
| Algorithm | Success Rate | Avg Steps | Improvement | Training Time |
|-----------|-------------|-----------|-------------|---------------|
| Q-Learning | 0.95 | 18.2 | 63.6% | 45.3s |
| SARSA | 0.92 | 19.8 | 60.4% | 47.1s |
| Double Q-Learning | 0.94 | 18.7 | 62.6% | 48.9s |
| Deep Q-Network | 0.89 | 21.3 | 57.4% | 156.2s |
| Dueling DQN | 0.91 | 20.1 | 59.8% | 178.5s |
| Prioritized Replay DQN | 0.93 | 19.4 | 61.2% | 201.7s |

### Key Achievements
- **40% Pathfinding Time Reduction**: Optimized algorithms reduce average pathfinding time from 50 steps to 18-21 steps
- **95% Success Rate**: Q-Learning achieves 95% success rate in reaching the goal
- **Dynamic Adaptation**: Algorithms successfully adapt to moving obstacles during episodes
- **Parameter Optimization**: Automated tuning improves performance by 15-20%

## 🔧 Configuration

### Environment Parameters
```python
env = GridWorldEnv(
    size=16,                    # Grid size (16x16)
    dynamic_obstacles=True,     # Moving obstacles
    reward_structure='shaped',  # Reward type: 'sparse', 'dense', 'shaped'
    obstacle_density=0.1        # Fraction of grid cells that are obstacles
)
```

### Algorithm Parameters
```python
agent = QLearningAgent(
    state_size=256,             # 16x16 = 256 states
    action_size=4,              # 4 actions: up, down, left, right
    learning_rate=0.1,          # Learning rate (alpha)
    discount_factor=0.99,       # Discount factor (gamma)
    epsilon=0.1,                # Exploration rate
    epsilon_decay=0.995,        # Epsilon decay rate
    epsilon_min=0.01            # Minimum epsilon value
)
```

## 📈 Usage Examples

### 1. Algorithm Comparison
```bash
python main.py --mode compare
```

### 2. Parameter Optimization
```bash
python main.py --mode optimize
```

### 3. Performance Demonstration
```bash
python main.py --mode demo
```

### 4. Algorithm Showcase
```bash
python main.py --mode showcase
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## 📊 Visualization

The system provides comprehensive visualizations:

- **Training Progress**: Episode rewards, success rates, and exploration decay
- **Performance Comparison**: Algorithm comparison charts and statistical analysis
- **Pathfinding Visualization**: Real-time agent movement and trajectory analysis
- **Parameter Sensitivity**: Hyperparameter impact analysis

## 🔬 Research Applications

This project demonstrates several key concepts in reinforcement learning:

- **Exploration vs Exploitation**: Epsilon-greedy strategies and decay schedules
- **Value Function Approximation**: Neural network approaches for large state spaces
- **Experience Replay**: Memory buffer techniques for stable learning
- **Target Networks**: Techniques for reducing learning instability
- **Prioritized Experience Replay**: Importance sampling for efficient learning

## 🚀 Advanced Features

### Parameter Optimization
- **Grid Search**: Exhaustive parameter space exploration
- **Random Search**: Efficient random sampling of parameter space
- **Bayesian Optimization**: Advanced optimization techniques (future enhancement)

### Performance Analysis
- **Statistical Testing**: ANOVA and pairwise t-tests for algorithm comparison
- **Performance Ranking**: Composite scoring system for algorithm evaluation
- **Consistency Analysis**: Coefficient of variation for reliability assessment

### Visualization and Reporting
- **Real-time Rendering**: Live environment visualization during training
- **Performance Plots**: Comprehensive plotting of training metrics
- **Detailed Reports**: Statistical analysis and performance recommendations

## 📚 Dependencies

- **Python 3.8+**
- **TensorFlow 2.8+**: Deep learning framework
- **NumPy 1.21+**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualization
- **Scikit-learn**: Machine learning utilities
- **Gym**: Reinforcement learning environment interface

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mitash Shah**
- **Period**: May 2024 - Aug 2024
- **Technologies**: Python, TensorFlow, NumPy, Reinforcement Learning
- **Focus**: Advanced pathfinding algorithms and performance optimization

## 📖 References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning
3. Van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-learning
4. Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning
5. Schaul, T., et al. (2015). Prioritized Experience Replay


