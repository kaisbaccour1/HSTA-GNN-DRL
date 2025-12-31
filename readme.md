# Joint Latency-Energy Aware Service Placement in Vehicular Edge Computing via Graph Neural Networks Predictions and Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red.svg)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-2.2.1-orange.svg)](https://www.dgl.ai/)
[![SUMO](https://img.shields.io/badge/SUMO-1.19.0-yellow.svg)](https://sumo.dlr.de/)

## Description

**HSTA-GNN+DRL** is a comprehensive framework for intelligent service placement in vehicular networks that combines:
- **Graph Neural Networks (GNNs)** for spatio-temporal service demand prediction
- **Deep Reinforcement Learning (DRL)** for optimal service placement decisions
- **SUMO** for realistic vehicular network simulation

The system addresses the challenge of efficiently placing compute-intensive services (like cooperative perception, platooning control, etc.) across edge nodes and cloud infrastructure in dynamic vehicular environments.

## Key Features

- **Realistic Simulation**: SUMO-based vehicular network simulation with Markov service demand models
- **Spatio-temporal GNN**: Heterogeneous GNN with temporal attention for service demand prediction
- **Multi-objective DRL**: Reinforcement learning agents optimized for energy, latency, or combined objectives
- **Comprehensive Evaluation**: Comparison of 6 deployment strategies with statistical analysis
- **Visual Analytics**: Detailed visualization of system state and energy/latency breakdowns

## Project Structure
```bash

HSTA-GNN+DRL/
├── src/ # Source code
│ ├── vehicular_network/ # SUMO simulation & graph generation
│ └── service_placement/ # GNN prediction + RL placement
│ ├── agents/ # RL agents
│ ├── environment/ # RL environment
│ ├── models/ # ML models
│ ├── training/ # Training routines
│ ├── evaluation/ # Evaluation framework
│ └── utils/ # Utilities & configs
├── data/ # Network data and generated graphs
├── models/ # Trained models and scripts
├── notebooks/ # Jupyter notebooks
└── experiment_plots/ # Experimental results (will be created after execution)
```

## Quick Start

### Prerequisites

- Python 3.9 or higher
- SUMO 1.19.0 or higher (for simulation)
- CUDA-capable GPU (recommended for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kaisbaccour1/HSTA-GNN+DRL.git
cd HSTA-GNN+DRL
```

2. **Create and activate a virtual environment**

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash

pip install -r requirements.txt
```
4. **Set up SUMO (if not already installed)**

```bash

# On Ubuntu/Debian
sudo apt-get install sumo sumo-tools sumo-doc

# On macOS
brew install sumo

# Or download from: https://sumo.dlr.de/docs/Downloads.php
```

### Basic Usage
1. **Generate simulation data**

```bash
python -m models.scripts.run_simulation
```

2. **Train prediction model**

```bash
python -m models.scripts.run_training_and_evaluation --train-prediction
```

3. **Train RL agents**

```bash
python -m models.scripts.run_training_and_evaluation --train-rl
```

4. **Train RL agents**

```bash
python -m models.scripts.run_training_and_evaluation --evaluate
```

**Complete Pipeline**

Run the entire pipeline:
```bash
python -m models.scripts.run_training_and_evaluation --train-prediction --train-rl --evaluate
```
### Troubleshooting and Alternative Execution

If you encounter issues running the Python scripts (.py files), Jupyter notebooks are available and work perfectly. They provide an interactive alternative to run the entire pipeline step by step.

**To run the notebooks**
```bash
# Launch Jupyter Notebook
jupyter notebook

# Or launch Jupyter Lab
jupyter lab
```

Then navigate to the notebooks/ directory and open the notebooks in order.

## Evaluation Strategies
The framework evaluates 6 deployment strategies:

1. Random - Random service deployment

2. Zero-Deployment - All services placed in cloud

3. Prediction-Follower - Follow GNN predictions directly

4. RL-Energy - RL optimized for energy consumption

5. RL-Latency - RL optimized for latency

6. RL-Combined - RL optimized for energy-latency trade-off




## Experiments
### Reproducing Experiments
To reproduce the experiments from the paper:

```bash


# 1. Generate data with specific parameters
python -m models.scripts.run_simulation \
    --vehicles 30 \
    --steps 200 \
    --edges 7

# 2. Run full experiment pipeline
python -m models.scripts.run_training_and_evaluation \
    --repetitions 30 \
    --episodes 100 \
    --reward-types energy_only latency_only energy_latency_combined

```

### Custom Experiments
Modify the configuration in src/service_placement/utils/config.py:

- Change number of services

- Adjust reward weights

- Modify network architecture

- Set different optimization objectives
















