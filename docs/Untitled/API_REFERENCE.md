# SPIDER API Reference

This document provides a comprehensive reference for the SPIDER API, including all public functions, classes, and modules.

## Table of Contents

- [Core Module (`spider.core`)](#core-module-spidercore)
- [I/O Module (`spider.io`)](#io-module-spiderio)
- [Analysis Module (`spider.analysis`)](#analysis-module-spideranalysis)
- [Utilities (`spider.utils`)](#utilities-spiderutils)
- [Main Entry Point (`SPIDER.py`)](#main-entry-point-spiderpy)

## Core Module (`spider.core`)

### Data Preparation

#### `prepare_input_dfs(params: dict) -> tuple`

Prepares input dataframes from CSV files for SPIDER processing.

**Parameters:**
- `params` (dict): Configuration dictionary containing file paths

**Returns:**
- `tuple`: (stations_df, dtimes_df, origins_df)

**Example:**
```python
from spider.core import prepare_input_dfs

with open('params.json', 'r') as f:
    params = json.load(f)

stations, dtimes, origins = prepare_input_dfs(params)
print(f"Loaded {len(origins)} events with {len(dtimes)} differential times")
```

### Location Pipeline

#### `locate_all(params: dict, origins: pd.DataFrame, dtimes: pd.DataFrame, model: torch.nn.Module, device: str, wandb_logger=None) -> None`

Runs the complete SPIDER location pipeline.

**Parameters:**
- `params` (dict): Configuration dictionary
- `origins` (pd.DataFrame): Event origins dataframe
- `dtimes` (pd.DataFrame): Differential times dataframe
- `model` (torch.nn.Module): EikoNet model
- `device` (str): Target device ('cuda' or 'cpu')
- `wandb_logger` (optional): Weights & Biases logger

**Example:**
```python
from spider.core import locate_all
from SPIDER import setup_nn_model

# Setup model and data
model = setup_nn_model(params, device)
stations, dtimes, origins = prepare_input_dfs(params)

# Run location
locate_all(params, origins, dtimes, model, device, wandb_logger)
```

### Modeling

#### `EikoNet(scale: float, vs: float = 3.3, vp: float = 6.0)`

Neural network model for travel time prediction.

**Parameters:**
- `scale` (float): Spatial scale factor
- `vs` (float): S-wave velocity (default: 3.3)
- `vp` (float): P-wave velocity (default: 6.0)

**Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass
- `EikonalPDE(x: torch.Tensor) -> torch.Tensor`: Eikonal equation evaluation

**Example:**
```python
from SPIDER import EikoNet

model = EikoNet(scale=200.0, vs=3.3, vp=6.0)
model.eval()
```

## I/O Module (`spider.io`)

### Sample Reading

#### `read_all_samples(params: dict, thin: int = 1, backend: str = 'pandas', device: str = None, event_ids: list = None) -> dict`

Reads MCMC samples from HDF5 file with optional thinning.

**Parameters:**
- `params` (dict): Configuration dictionary
- `thin` (int): Thinning factor (default: 1)
- `backend` (str): Backend to use ('pandas' or 'torch', default: 'pandas')
- `device` (str): Target device for torch backend (default: None)
- `event_ids` (list): Specific event IDs to read (default: None)

**Returns:**
- `dict`: Dictionary containing sample arrays

**Example:**
```python
from spider.io.samples import read_all_samples

# Read all samples
samples = read_all_samples(params)

# Read with thinning
samples_thinned = read_all_samples(params, thin=5)

# Read specific events
event_ids = ['evt_001', 'evt_002']
selective_samples = read_all_samples(params, event_ids=event_ids)
```

#### `read_samples_smart(params: dict, event_ids: list = None, auto_migrate: bool = True) -> dict`

Smart sample reading that automatically detects file format.

**Parameters:**
- `params` (dict): Configuration dictionary
- `event_ids` (list): Specific event IDs to read (default: None)
- `auto_migrate` (bool): Automatically migrate legacy format (default: True)

**Returns:**
- `dict`: Dictionary containing sample arrays

### Sample Writing

#### `save_samples_batch(samples: dict, params: dict, batch_num: int) -> None`

Saves a batch of MCMC samples to HDF5 file.

**Parameters:**
- `samples` (dict): Sample dictionary
- `params` (dict): Configuration dictionary
- `batch_num` (int): Batch number

### Checkpointing

#### `save_checkpoint(state: dict, params: dict, epoch: int) -> None`

Saves optimization checkpoint.

**Parameters:**
- `state` (dict): Current optimization state
- `params` (dict): Configuration dictionary
- `epoch` (int): Current epoch number

#### `load_checkpoint(params: dict) -> tuple`

Loads optimization checkpoint.

**Parameters:**
- `params` (dict): Configuration dictionary

**Returns:**
- `tuple`: (state, epoch) or (None, 0) if no checkpoint found

## Analysis Module (`spider.analysis`)

### Summary Statistics

#### `compute_cat_dd_and_xyz(event_samples: dict, burn_in: int = 0, thin: int = 1) -> EventSamplesSummary`

Computes summary statistics from MCMC samples.

**Parameters:**
- `event_samples` (dict): Sample dictionary from `read_all_samples`
- `burn_in` (int): Number of samples to discard (default: 0)
- `thin` (int): Thinning factor (default: 1)

**Returns:**
- `EventSamplesSummary`: Object containing summary statistics

**Example:**
```python
from spider.analysis import compute_cat_dd_and_xyz

# Compute summary with burn-in
summary = compute_cat_dd_and_xyz(samples, burn_in=100, thin=5)

# Access results
print(f"Located {summary.X.shape[0]} events")
print(f"Mean longitude: {summary.X[:, 0].mean():.4f}")
print(f"Longitude std: {summary.X[:, 0].std():.4f}")
```

### EventSamplesSummary Class

Container for location results and uncertainty estimates.

**Attributes:**
- `X` (np.ndarray): Event coordinates [longitude, latitude, depth, time]
- `longitude` (np.ndarray): Longitude samples
- `latitude` (np.ndarray): Latitude samples
- `depth` (np.ndarray): Depth samples
- `delta_t` (np.ndarray): Time samples
- `event_ids` (list): Event identifiers

**Methods:**
- `compute(burn_in: int = 0, thin: int = 1) -> EventSamplesSummary`: Compute summary statistics
- `save_catalog(filename: str) -> None`: Save results to CSV catalog
- `get_uncertainty(event_idx: int) -> dict`: Get uncertainty estimates for specific event

**Example:**
```python
# Get uncertainty for first event
uncertainty = summary.get_uncertainty(0)
print(f"Longitude uncertainty: ±{uncertainty['longitude_std']:.4f}°")
print(f"Depth uncertainty: ±{uncertainty['depth_std']:.2f} km")

# Save results
summary.save_catalog("final_locations.csv")
```

## Utilities (`spider.utils`)

### Weights & Biases Integration

#### `init_wandb_if_enabled(params: dict) -> wandb.Run or None`

Initializes Weights & Biases logging if enabled.

**Parameters:**
- `params` (dict): Configuration dictionary

**Returns:**
- `wandb.Run` or `None`: W&B run object if enabled, None otherwise

**Example:**
```python
from spider.utils import init_wandb_if_enabled

# Initialize wandb if enabled in params
wandb_logger = init_wandb_if_enabled(params)

# Use in location pipeline
locate_all(params, origins, dtimes, model, device, wandb_logger)
```

### Model Setup

#### `setup_nn_model(params: dict, device: str) -> torch.nn.Module`

Loads and configures the neural network model.

**Parameters:**
- `params` (dict): Configuration dictionary containing 'model_file'
- `device` (str): Target device

**Returns:**
- `torch.nn.Module`: Loaded and configured model

**Example:**
```python
from SPIDER import setup_nn_model

device = params["devices"][0]
model = setup_nn_model(params, device)
model.eval()
```

## Main Entry Point (`SPIDER.py`)

### Command Line Interface

```bash
python SPIDER.py [parameter_file]
```

**Parameters:**
- `parameter_file` (str): Path to JSON configuration file

**Example:**
```bash
python SPIDER.py params.json
```

### Main Functions

#### `main(spider_pfile: str) -> None`

Main execution function.

**Parameters:**
- `spider_pfile` (str): Path to parameter file

**Example:**
```python
from SPIDER import main

main("params.json")
```

## Configuration Parameters

### EikoNet Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `velmod_file` | string | Path to velocity model CSV | Required |
| `lon_min`, `lat_min` | float | Spatial domain bounds | Required |
| `z_min`, `z_max` | float | Depth bounds (km) | Required |
| `scale` | float | Spatial scale factor | Required |
| `model_file` | string | Output model path | Required |
| `train_batch_size` | int | Training batch size | 512 |
| `val_batch_size` | int | Validation batch size | 10000 |
| `n_train` | int | Training samples | 1000000 |
| `n_test` | int | Test samples | 2000000 |
| `n_epochs` | int | Training epochs | 1000 |
| `lr` | float | Learning rate | 1e-3 |

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtime_file` | str | Path to differential times CSV |
| `catalog_infile` | str | Path to input event catalog |
| `catalog_outfile` | str | Path for output catalog |
| `samples_outfile` | str | Path for HDF5 samples file |
| `model_file` | str | Path to EikoNet model file |

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

### Weights & Biases Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `use_wandb` | bool | Enable/disable wandb logging |
| `wandb_project_name` | str | W&B project name |
| `wandb_run_name` | str | W&B run name |

## Data Formats

### Input CSV Formats

**Event Catalog** (`catalog_infile`):
```csv
event_id,longitude,latitude,depth,origin_time
evt_001,-117.5,34.2,10.5,2023-01-01T12:00:00
evt_002,-117.6,34.3,12.1,2023-01-01T12:05:00
```

**Differential Times** (`dtime_file`):
```csv
event1_id,event2_id,station,phase,dt_obs,dt_err
evt_001,evt_002,STA1,P,0.5,0.1
evt_001,evt_002,STA1,S,1.2,0.15
```

### Output Formats

**HDF5 Samples** (`samples_outfile`):
- Optimized format for fast reading
- Contains all MCMC samples
- Supports thinning and selective reading

**CSV Catalog** (`catalog_outfile`):
```csv
event_id,longitude,latitude,depth,origin_time,longitude_std,latitude_std,depth_std,time_std
evt_001,-117.5001,34.2001,10.5001,2023-01-01T12:00:00.001,0.001,0.001,0.1,0.001
```

## Error Handling

### Common Exceptions

- `FileNotFoundError`: Input files not found
- `ValueError`: Invalid parameter values
- `RuntimeError`: GPU memory issues
- `KeyError`: Missing required parameters

### Error Recovery

- **Checkpointing**: Automatic recovery from interruptions
- **Memory Management**: Thinning for large datasets
- **Validation**: Parameter validation before execution

## Performance Considerations

### GPU Memory Optimization

- Use `thin` parameter for large datasets
- Adjust batch sizes based on GPU memory
- Monitor memory usage with wandb

### Parallel Processing

- Multi-GPU support via `devices` parameter
- Parallel HDF5 reading for large files
- Optimized data loading pipelines

---

For more information, see the [README](README.md) and [example scripts](example_*.py).
