# Reverso

A comprehensive Python framework for reverse engineering and neural network analysis, providing tools for environment simulation, agent training, and interaction network analysis (Year 2023).

A Machine Learning Approach to Simulate Gene Expression and Infer Gene Regulatory Networks
https://doi.org/10.3390/e25081214

Deep Learning and Metaheuristic for Multivariate Time-Series Forecasting 
https://doi.org/10.1007/978-3-031-42529-5_24

A Novel Reverse Engineering Approach for Gene Regulatory Networks 
https://doi.org/10.1007/978-3-031-21127-0_26

Optimizing Multi-variable Time Series Forecasting Using Metaheuristics 
https://doi.org/10.1007/978-3-031-26504-4_8

## Overview

Reverso is designed to facilitate reverse engineering tasks through machine learning and network analysis. It provides a unified framework for:

- **Environment Simulation**: Create and manage complex environments for testing and analysis
- **AI Agent Training**: Train various types of neural network agents (MLP, CNN, LSTM, etc.)
- **Interaction Network Analysis**: Analyze and visualize complex interaction patterns
- **Data Collection and Processing**: Handle experimental data and datasets
- **Perturbation Analysis**: Study system behavior under various perturbations

## Features

### Core Components

- **Environment Management**: Sophisticated environment simulation and state management
- **Agent Framework**: Multiple agent types with configurable architectures
- **Network Analysis**: Advanced tools for interaction network analysis
- **Data Handling**: Comprehensive data collection and processing capabilities
- **Utilities**: Rich set of utility functions for metrics, workspace management, and more

### Supported Agent Types

- **MLP Agent**: Multi-layer perceptron networks
- **CNN Agent**: Convolutional neural networks
- **LSTM Agent**: Long short-term memory networks
- **Dynamic Agent**: Dynamically configurable architectures
- **Neural Network Agent**: General-purpose neural network implementation

### Analysis Tools

- **Signal Perturbation**: Advanced perturbation analysis functions
- **Environment Analysis**: Comprehensive environment state analysis
- **Interaction Networks**: Network topology analysis and visualization
- **Metrics Collection**: Detailed performance and behavior metrics

## Installation

### Requirements

- Python >= 3.10
- TensorFlow >= 2.0
- NumPy, Pandas, Scikit-learn
- NetworkX for graph analysis
- Matplotlib and Seaborn for visualization

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd reverso

# Install in development mode
pip install -e .

# Or use the provided installation scripts
./install.sh        # Standard installation
./install-debug.sh  # Development installation
```

### Dependencies

The framework requires several key dependencies:

```
tensorflow>=2.0
tensorflow-text>=2.0
tf-models-official>=2.0
networkx>=3.1
scikit-learn==1.2.2
pandas==2.0.3
matplotlib==3.7.4
numpy==1.24.3
```

See `pyproject.toml` for the complete dependency list.

## Quick Start

```python
import reverso
from reverso import Reverso, WorkspaceInfo, Environment

# Initialize workspace
workspace = WorkspaceInfo(session_id="example")
reverso_instance = Reverso(workspace)

# Create environment
env = Environment()

# Configure and train agents
agent_config = reverso.AgentConfiguration()
agent = reverso.AgentBuilder.build(agent_config)

# Analyze interaction networks
network = reverso.InteractionNetwork()
analysis = reverso.EnvironmentAnalysis()
```

## Project Structure

```
src/reverso/
├── __init__.py           # Main package initialization
├── reverso.py           # Core Reverso class
├── reversoconfig.py     # Configuration management
├── agents/              # AI agent implementations
│   ├── agent.py
│   ├── agentbuilder.py
│   ├── cnnagent.py
│   ├── lstmagent.py
│   └── mlpagent.py
├── core/                # Core functionality
│   ├── environment.py   # Environment simulation
│   ├── network.py       # Interaction networks
│   ├── analysis.py      # Analysis tools
│   └── pertubations.py  # Perturbation functions
├── data/                # Data handling
│   ├── collection.py    # Data collection
│   └── experiments.py   # Experiment management
└── utilities/           # Utility functions
    ├── metrics.py       # Performance metrics
    ├── workspaceinfo.py # Workspace management
    └── ...
```

## Usage Examples

### Environment Setup

```python
from reverso import Environment, EnvironmentConfiguration

# Create and configure environment
env_config = EnvironmentConfiguration()
env = Environment(env_config)
```

### Agent Training

```python
from reverso import AgentBuilder, AgentConfiguration

# Configure agent
config = AgentConfiguration(
    agent_type="MLP",
    hidden_layers=[64, 32],
    activation="relu"
)

# Build and train agent
agent = AgentBuilder.build(config)
train_option = AgentTrainOption(epochs=100, batch_size=32)
agent.train(training_data, train_option)
```

### Network Analysis

```python
from reverso import InteractionNetwork, InteractionNetworkHelper

# Create interaction network
network = InteractionNetwork()
helper = InteractionNetworkHelper()

# Analyze network properties
analysis_result = helper.analyze(network)
```
