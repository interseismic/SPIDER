# SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

SPIDER is a state-of-the-art Python toolkit for probabilistic earthquake location using deep learning and MCMC sampling. It combines neural network-based travel time prediction with scalable Bayesian inference to provide accurate earthquake locations with uncertainty quantification.

## 🌟 Key Features

- **Neural Network Travel Time Prediction**: Uses EikoNet for fast travel time calculations
- **Multi-Phase MCMC Sampling**: Four-phase optimization pipeline for robust location estimates
- **GPU Acceleration**: Full CUDA support for high-performance computing
- **Weights & Biases Integration**: Experiment tracking and metric visualization
- **HDF5 Sample Storage**: Efficient storage and retrieval of MCMC samples
- **MCMC Chain Thinning**: Memory-efficient analysis of large sample sets
- **Checkpointing**: Robust recovery from interruptions
- **Modular Architecture**: Clean separation of core algorithms and utilities

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- PyTorch 1.8+ with CUDA support

### Install SPIDER

```bash
# Clone the repository
git clone https://github.com/your-username/SPIDER.git
cd SPIDER

# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e .[wandb,dev]
```

### Verify Installation

```bash
python -c "import spider; print('SPIDER installed successfully!')"
```

## ⚡ Quick Start

### 1. Train EikoNet Model (Required)

Before running SPIDER, you need a trained EikoNet model for travel time prediction.

**Create EikoNet configuration** (`eikonet.json`):
```json
{
    "velmod_file": "path/to/velmod.csv",
    "lon_min": -117.5,
    "lat_min": 33.0,
    "z_min": -5.0,
    "z_max": 80.0,
    "scale": 400.0,
    "model_file": "path/to/model.pt",
    "train_batch_size": 512,
    "val_batch_size": 10000,
    "n_train": 1000000,
    "n_test": 2000000,
    "n_epochs": 1000,
    "lr": 1e-3
}
```

**Train the model**:
```bash
# Download and run EikoNet training script
python eikonet_train.py eikonet.json
```

**Velocity model format** (`velmod.csv`):
```csv
depth,vs,vp
-5.0,3.3,6.0
0.0,3.3,6.0
5.0,3.4,6.1
10.0,3.5,6.2
...
```

### 2. Prepare Your Data

SPIDER requires two main input files:

**Event Catalog** (`events.csv`):
```csv
event_id,longitude,latitude,depth,origin_time
evt_001,-117.5,34.2,10.5,2023-01-01T12:00:00
evt_002,-117.6,34.3,12.1,2023-01-01T12:05:00
```

**Differential Times** (`dtimes.csv`):
```csv
event1_id,event2_id,network,station,phase,dt
evt_001,evt_002,NET1,STA1,P,0.5
evt_001,evt_002,NET1,STA1,S,1.2
```

### 2. Create Configuration File

Create `params.json`:

```json
{
    "dtime_file": "dtimes.csv",
    "catalog_infile": "events.csv",
    "catalog_outfile": "SPIDER_out.cat",
    "samples_outfile": "SPIDER_samples.h5",
    "model_file": "path/to/eikonet_model.pt",
    
    "lon_min": -120.0,
    "lat_min": 35.0,
    "z_min": -5.0,
    "z_max": 50.0,
    "scale": 200.0,
    
    "phase1_epochs": 1000,
    "phase2_epochs": 200,
    "phase3_epochs": 500,
    "phase4_epochs": 10000,
    
    "lr_warmup": 2e-3,
    "lr_sgld": 2e-3,
    "batch_size_warmup": 10000,
    "batch_size_sgld": 10000,
    
    "devices": [0]
}
```

### 3. Run SPIDER

```bash
python SPIDER.py params.json
```

### 4. Analyze Results

```python
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz

# Load parameters
with open('params.json', 'r') as f:
    params = json.load(f)

# Read samples
samples = read_all_samples(params)

# Compute summary statistics
summary = compute_cat_dd_and_xyz(samples, burn_in=100)

# Access results
print(f"Located {summary.X.shape[0]} events")
print(f"Mean longitude: {summary.X[:, 0].mean():.4f}")
```

## ⚙️ Configuration

### Core Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `dtime_file` | string | Path to differential times CSV | Required |
| `catalog_infile` | string | Path to input event catalog | Required |
| `catalog_outfile` | string | Path for output catalog | Required |
| `samples_outfile` | string | Path for HDF5 samples file | Required |
| `model_file` | string | Path to EikoNet model file | Required |

### Spatial Domain

| Parameter | Type | Description |
|-----------|------|-------------|
| `lon_min`, `lon_max` | float | Longitude bounds |
| `lat_min`, `lat_max` | float | Latitude bounds |
| `z_min`, `z_max` | float | Depth bounds (km) |
| `scale` | float | Spatial scale factor |

### Optimization Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `phase1_epochs` | int | MAP estimation epochs |
| `phase2_epochs` | int | Preconditioned drift epochs |
| `phase3_epochs` | int | Noise ramp epochs |
| `phase4_epochs` | int | Full SGLD sampling epochs |
| `lr_warmup` | float | Adam learning rate |
| `lr_sgld` | float | SGLD learning rate |
| `batch_size_warmup` | int | Batch size for warmup |
| `batch_size_sgld` | int | Batch size for SGLD |

### Uncertainty Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `phase_unc` | list | Phase uncertainty [P, S] |
| `prior_event_std` | list | Event prior std [x, y, z, t] |
| `prior_centroid_std` | list | Centroid prior std [x, y, z, t] |

### Hardware Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `devices` | list | GPU device IDs to use |

## 📚 Usage Examples

### Basic Location

```python
import json
from SPIDER import setup_nn_model
from spider.core import prepare_input_dfs, locate_all
from spider.utils import init_wandb_if_enabled

# Load parameters
with open('params.json', 'r') as f:
    params = json.load(f)

# Setup model
device = params["devices"][0]
model = setup_nn_model(params, device)

# Prepare data
stations, dtimes, origins = prepare_input_dfs(params)

# Initialize wandb (optional)
wandb_logger = init_wandb_if_enabled(params)

# Run location
locate_all(params, origins, dtimes, model, device, wandb_logger)
```

### With Weights & Biases

```json
{
    "use_wandb": true,
    "wandb_project_name": "spider-location",
    "wandb_run_name": "experiment_001",
    "sgld_alg": "psgld"
}
```

### MCMC Chain Thinning

```python
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz

# Read samples with thinning (keep every 5th sample)
samples = read_all_samples(params, thin=5)

# Compute summary with additional thinning
summary = compute_cat_dd_and_xyz(samples, burn_in=100, thin=2)
```

### Checkpointing

```json
{
    "checkpoint_dir": "checkpoints/",
    "checkpoint_interval": 100,
    "min_samples_to_save": 10
}
```

## 🔧 Advanced Features

### EikoNet Model Training

SPIDER uses EikoNet, a neural network for fast travel time prediction. You must train this model before running SPIDER.

#### Training Configuration

**EikoNet JSON Parameters** (`eikonet.json`):

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `velmod_file` | string | Path to velocity model CSV | `"velmod.csv"` |
| `lon_min`, `lat_min` | float | Spatial domain bounds | `-117.5, 33.0` |
| `z_min`, `z_max` | float | Depth bounds (km) | `-5.0, 80.0` |
| `scale` | float | Spatial scale factor | `400.0` |
| `model_file` | string | Output model path | `"model.pt"` |
| `train_batch_size` | int | Training batch size | `512` |
| `val_batch_size` | int | Validation batch size | `10000` |
| `n_train` | int | Training samples | `1000000` |
| `n_test` | int | Test samples | `2000000` |
| `n_epochs` | int | Training epochs | `1000` |
| `lr` | float | Learning rate | `1e-3` |

#### Velocity Model Format

Create a CSV file with depth-dependent velocities:

```csv
depth,vs,vp
-5.0,3.3,6.0
0.0,3.3,6.0
5.0,3.4,6.1
10.0,3.5,6.2
15.0,3.6,6.3
20.0,3.7,6.4
...
```

#### Training Process

```bash
# 1. Prepare velocity model
# Create velmod.csv with your regional velocity structure

# 2. Configure training
# Edit eikonet.json with your parameters

# 3. Train the model
python eikonet_train.py eikonet.json

# 4. Verify training
# Check that model.pt was created successfully
```

#### Training Tips

- **Domain size**: Ensure spatial bounds cover your study area
- **Depth range**: Include all relevant depths for your earthquakes
- **Velocity model**: Use region-specific velocity structure
- **Training samples**: More samples = better accuracy (but longer training)
- **Batch size**: Adjust based on GPU memory
- **Learning rate**: Start with 1e-3, reduce if training is unstable

For detailed EikoNet training instructions, see [docs/EIKONET_TRAINING.md](docs/EIKONET_TRAINING.md).

### Weights & Biases Integration

SPIDER supports comprehensive experiment tracking with Weights & Biases:

- **Automatic metric logging** across all optimization phases
- **Hyperparameter tracking** for reproducibility
- **Real-time visualization** of convergence
- **Experiment comparison** and collaboration

See [WANDB_INTEGRATION.md](WANDB_INTEGRATION.md) for detailed documentation.

### MCMC Chain Thinning

For memory-efficient analysis of large sample sets:

- **GPU memory optimization** during data loading
- **Configurable thinning factors** for different analysis stages
- **Consistent thinning** across all coordinate fields

See [THINNING_FUNCTIONALITY.md](THINNING_FUNCTIONALITY.md) for details.

### HDF5 Sample Storage

Efficient storage and retrieval of MCMC samples:

- **Optimized format** for fast reading
- **Parallel processing** support
- **Selective event loading**
- **Automatic format migration**

### Checkpointing

Robust recovery from interruptions:

- **Automatic checkpointing** at configurable intervals
- **Resume capability** from any checkpoint
- **Sample preservation** during interruptions

## 📖 API Reference

### Core Functions

#### `spider.core.locate.locate_all()`
Main location function that runs the complete SPIDER pipeline.

#### `spider.core.data.prepare_input_dfs()`
Prepares input dataframes from CSV files.

#### `spider.io.samples.read_all_samples()`
Reads MCMC samples from HDF5 file with optional thinning.

#### `spider.analysis.compute_cat_dd_and_xyz()`
Computes summary statistics from MCMC samples.

### Classes

#### `EikoNet`
Neural network model for travel time prediction.

#### `EventSamplesSummary`
Container for location results and uncertainty estimates.

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce batch size
"batch_size_warmup": 5000,
"batch_size_sgld": 5000

# Use MCMC thinning
samples = read_all_samples(params, thin=5)
```

**Slow Performance**
```bash
# Enable parallel processing
"num_workers": 4

# Use optimized HDF5 format
# (automatic with newer versions)
```

**Model Loading Errors**
```python
# Ensure model file path is correct
# Check PyTorch version compatibility
torch.load(model_file, map_location='cpu')
```

### Getting Help

1. Check the [example scripts](example_*.py)
2. Review the [notebooks](notebooks/) for usage patterns
3. Examine [configuration examples](*.json)
4. Open an issue with detailed error information

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/your-username/SPIDER.git
cd SPIDER
pip install -e .[dev]

# Run tests
python -m pytest tests/

# Format code
black spider/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use SPIDER in your research, please cite:

```bibtex
@software{spider2024,
  title={SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation},
  author={Your Name and Collaborators},
  year={2024},
  url={https://github.com/your-username/SPIDER}
}
```

## 🙏 Acknowledgments

- Built on PyTorch and modern deep learning techniques
- Inspired by EikoNet for travel time prediction
- Uses HDF5 for efficient data storage
- Integrates with Weights & Biases for experiment tracking

---

**For questions and support, please open an issue on GitHub.**
