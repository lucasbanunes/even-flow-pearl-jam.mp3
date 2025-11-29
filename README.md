# Even Flow Pearl Jam 🎵

A comprehensive repository with training and evaluation pipelines for flow-based generative models including Continuous Normalizing Flows (CNFs), Neural ODEs, Real NVP, and other flow-based architectures.

## 🚀 Quick Start

### Prerequisites

- Python 3.12.12
- [uv](https://github.com/astral-sh/uv) for package management

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd even-flow-pearl-jam.mp3
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Using the virtual environment:
```bash
uv run <desired-command>
```
Calling uv before the command ensures it runs within the managed repository environment.

## 📁 Repository Structure

```
even-flow-pearl-jam.mp3/
├── cli.py                          # Main CLI entry point
├── pyproject.toml                  # Project configuration and dependencies
├── even_flow/                      # Main package
│   ├── __init__.py
│   ├── jobs.py                     # Base job classes for training
│   ├── metrics.py                  # Custom metrics
│   ├── mlflow.py                   # MLflow integration utilities
│   ├── plotting.py                 # Visualization utilities
│   ├── pydantic.py                 # Pydantic base models
│   ├── torch.py                    # PyTorch utilities
│   ├── utils.py                    # General utilities
│   ├── moons/                      # Two moons dataset experiments
│   │   ├── cli.py                  # Moons-specific CLI commands
│   │   ├── dataset.py              # Moons dataset implementation
│   │   ├── jobs.py                 # Training jobs for moons
│   │   └── models.py               # Moons-specific model implementations
│   ├── spirals/                    # Spiral dataset experiments
│   │   ├── dataset.py              # Spirals dataset implementation
│   │   └── jobs.py                 # Training jobs for spirals
│   ├── mnist/                      # MNIST dataset experiments
│   │   ├── dataset.py              # MNIST dataset wrapper
│   │   └── jobs.py                 # MNIST training jobs
│   └── models/                     # Model implementations
│       ├── cnf.py                  # Continuous Normalizing Flows
│       ├── lightning.py            # PyTorch Lightning modules utilities
│       ├── mlp.py                  # Multi-layer perceptrons
│       ├── neuralode.py            # Neural ODE implementations
│       └── real_nvp.py             # Real NVP implementation
├── tests/                          # Test suite
└── *.ipynb                         # Jupyter notebooks for exploration
```

## 🛠️ Usage

The repository uses a CLI built with [Typer](https://typer.tiangolo.com/) for running training and evaluation jobs. Each job is configured via YAML files.

### Available Commands

#### Main CLI
```bash
uv run python cli.py --help
```

#### Moons Dataset Experiments

Train flow-based models on the two moons classification dataset:

```bash
# Neural ODE with time-embedded MLP
uv run python cli.py moons time-embedding-mlp-neural-ode --config config.yaml

# Continuous Normalizing Flow with exact trace
uv run python cli.py moons time-embedding-mlp-cnf --config config.yaml

# CNF with Hutchinson trace estimator
uv run python cli.py moons time-embedding-mlp-cnf-hutchingson --config config.yaml

# Real NVP normalizing flow
uv run python cli.py moons real-nvp --config config.yaml
```

### Configuration Files

Each training job requires a YAML configuration file. See `tests/data/test_moons/` for example configurations:

```yaml
# Example configuration for Neural ODE
datamodule:
  batch_size: 32
model:
  input_shape: [2]
  vector_field:
    input_dims: 2
    time_embed_dims: 3
    time_embed_freq: 10
    neurons_per_layer:
      - 13
      - 13
      - 2
    activations:
      - tanh
      - tanh
      - linear
  monitor: val_loss
  mode: min
  max_epochs: 3
```

## 🏗️ Architecture

### Core Components

#### 1. **Datasets**
- **Moons**: Two interleaving half-circles for non-linear classification
- **Spirals**: Parametric spiral trajectories for time series modeling
- **MNIST**: Handwritten digit classification

#### 2. **Models**
- **Neural ODEs**: Continuous-depth neural networks using ODE solvers
- **Continuous Normalizing Flows (CNFs)**: Invertible neural networks with continuous change of variables
- **Real NVP**: Real-valued Non-Volume Preserving transformations
- **Time-Embedded MLPs**: Multi-layer perceptrons with sinusoidal time embeddings

#### 3. **Training Pipeline**
- **PyTorch Lightning**: Training orchestration and distributed computing
- **MLflow**: Experiment tracking, model registry, and metrics logging
- **Pydantic**: Configuration validation and serialization
- **Typer**: Command-line interface generation

### Key Features

- **Modular Design**: Each component (dataset, model, job) is self-contained
- **Configuration-Driven**: All experiments defined via YAML files
- **Reproducible**: Deterministic random seeds and version pinning
- **Observable**: Comprehensive logging with MLflow integration
- **Type-Safe**: Full type annotations with Pydantic validation

## 📊 Experiment Tracking

The repository uses MLflow for experiment tracking:

1. **Metrics**: Training/validation losses, accuracies, NFE counts
2. **Parameters**: All model and training hyperparameters
3. **Artifacts**: Model checkpoints, plots, configuration files
4. **Models**: Serialized models with automatic versioning

Access the MLflow UI:
```bash
uv run mlflow ui
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
uv run pytest

# Run specific test module
uv run pytest tests/test_moons.py

# Run with verbose output
uv run pytest -v
```

## 📚 Notebooks

The repository includes several Jupyter notebooks for exploration:

- `01_neural_ode_2d.ipynb`: Introduction to Neural ODEs
- `02_cnf_trace_computation.ipynb`: CNF trace computation methods

Start JupyterLab:
```bash
uv run jupyter lab
```

## 🔧 Development

### Package Management

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management:

```bash
# Add a new dependency
uv add numpy

# Add development dependency
uv add --group dev pytest

# Update dependencies
uv sync
```

### Project Structure Philosophy

The codebase follows a domain-driven design:

- Each dataset (moons, spirals, mnist) has its own module
- Models are shared across domains but can have dataset-specific implementations
- Jobs orchestrate training and are dataset/model specific
- Configuration is handled via Pydantic models with YAML serialization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `uv run pytest`
5. Submit a pull request

## 📄 License

See [LICENSE](LICENSE) for details.

## 🎵 About the Name

"Even Flow" is a reference to the Pearl Jam song, one of the favorite songs of the repository creator. Just as the song flows seamlessly, this repository aims to provide a smooth experience for experimenting with flow-based generative models. Enjoy the rhythm of machine learning!
