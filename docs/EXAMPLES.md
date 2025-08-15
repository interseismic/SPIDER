# SPIDER Examples Guide

This guide provides comprehensive examples of how to use SPIDER for various earthquake location scenarios.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Advanced Configuration](#advanced-configuration)
- [Data Analysis](#data-analysis)
- [Performance Optimization](#performance-optimization)
- [Real-World Scenarios](#real-world-scenarios)
- [Troubleshooting Examples](#troubleshooting-examples)

## Basic Usage

### Simple Location Example

**File: `simple_location.py`**

```python
#!/usr/bin/env python3
"""
Simple example of running SPIDER for earthquake location.
"""

import json
import sys
from SPIDER import setup_nn_model
from spider.core import prepare_input_dfs, locate_all
from spider.utils import init_wandb_if_enabled

def main():
    # Load parameters
    with open('params.json', 'r') as f:
        params = json.load(f)
    
    # Setup model
    device = params["devices"][0]
    model = setup_nn_model(params, device)
    
    # Prepare data
    print("Preparing input dataset...")
    stations, dtimes, origins = prepare_input_dfs(params)
    
    print(f"Dataset has {origins.shape[0]} events with {dtimes.shape[0]} dtimes")
    
    # Initialize wandb (optional)
    wandb_logger = init_wandb_if_enabled(params)
    
    # Run location
    print("Running SPIDER...")
    locate_all(params, origins, dtimes, model, device, wandb_logger)
    
    # Finish wandb run
    if wandb_logger:
        wandb_logger.finish()
    
    print("Location complete!")

if __name__ == "__main__":
    main()
```

**Configuration: `params_simple.json`**

```json
{
    "dtime_file": "data/dtimes.csv",
    "catalog_infile": "data/events.csv",
    "catalog_outfile": "results/SPIDER_out.cat",
    "samples_outfile": "results/SPIDER_samples.h5",
    "model_file": "models/eikonet_model.pt",
    
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

### Command Line Usage

```bash
# Run with parameter file
python SPIDER.py params_simple.json

# Run with custom configuration
python SPIDER.py my_params.json
```

## Advanced Configuration

### With Weights & Biases Integration

**File: `wandb_example.py`**

```python
#!/usr/bin/env python3
"""
Example with Weights & Biases integration for experiment tracking.
"""

import json
import wandb
from SPIDER import setup_nn_model
from spider.core import prepare_input_dfs, locate_all
from spider.utils import init_wandb_if_enabled

def main():
    # Load parameters with wandb enabled
    with open('params_wandb.json', 'r') as f:
        params = json.load(f)
    
    # Setup model
    device = params["devices"][0]
    model = setup_nn_model(params, device)
    
    # Prepare data
    stations, dtimes, origins = prepare_input_dfs(params)
    
    # Add dataset info to params for wandb
    params["total_events"] = origins.shape[0]
    params["total_dtimes"] = dtimes.shape[0]
    
    # Initialize wandb
    wandb_logger = init_wandb_if_enabled(params)
    
    # Run location with logging
    locate_all(params, origins, dtimes, model, device, wandb_logger)
    
    # Log final results
    if wandb_logger:
        wandb_logger.log({
            "final_events_located": origins.shape[0],
            "final_dtimes_processed": dtimes.shape[0]
        })
        wandb_logger.finish()

if __name__ == "__main__":
    main()
```

**Configuration: `params_wandb.json`**

```json
{
    "dtime_file": "data/dtimes.csv",
    "catalog_infile": "data/events.csv",
    "catalog_outfile": "results/SPIDER_out.cat",
    "samples_outfile": "results/SPIDER_samples.h5",
    "model_file": "models/eikonet_model.pt",
    
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
    
    "devices": [0],
    
    "use_wandb": true,
    "wandb_project_name": "spider-location",
    "wandb_run_name": "experiment_001",
    
    "sgld_alg": "psgld",
    "sgld_beta1": 0.9,
    "sgld_beta2": 0.99
}
```

### With Checkpointing

**Configuration: `params_checkpoint.json`**

```json
{
    "dtime_file": "data/dtimes.csv",
    "catalog_infile": "data/events.csv",
    "catalog_outfile": "results/SPIDER_out.cat",
    "samples_outfile": "results/SPIDER_samples.h5",
    "model_file": "models/eikonet_model.pt",
    
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
    
    "devices": [0],
    
    "checkpoint_dir": "checkpoints/",
    "checkpoint_interval": 100,
    "min_samples_to_save": 10,
    
    "save_every_n": 100
}
```

## Data Analysis

### Reading and Analyzing Results

**File: `analyze_results.py`**

```python
#!/usr/bin/env python3
"""
Example of reading and analyzing SPIDER results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz

def analyze_results():
    # Load parameters
    with open('params.json', 'r') as f:
        params = json.load(f)
    
    # Read samples with different thinning factors
    print("Reading samples...")
    
    # Full dataset
    samples_full = read_all_samples(params, thin=1)
    
    # Thinned dataset for memory efficiency
    samples_thinned = read_all_samples(params, thin=5)
    
    # Compute summary statistics
    print("Computing summary statistics...")
    summary = compute_cat_dd_and_xyz(samples_thinned, burn_in=100, thin=1)
    
    # Access results
    print(f"Located {summary.X.shape[0]} events")
    print(f"Mean longitude: {summary.X[:, 0].mean():.4f}°")
    print(f"Mean latitude: {summary.X[:, 1].mean():.4f}°")
    print(f"Mean depth: {summary.X[:, 2].mean():.2f} km")
    
    # Get uncertainty estimates
    for i in range(min(5, summary.X.shape[0])):
        uncertainty = summary.get_uncertainty(i)
        print(f"Event {i}:")
        print(f"  Longitude: {summary.X[i, 0]:.4f}° ± {uncertainty['longitude_std']:.4f}°")
        print(f"  Latitude: {summary.X[i, 1]:.4f}° ± {uncertainty['latitude_std']:.4f}°")
        print(f"  Depth: {summary.X[i, 2]:.2f} km ± {uncertainty['depth_std']:.2f} km")
    
    # Save results
    summary.save_catalog("final_locations.csv")
    
    return summary

def plot_results(summary):
    """Plot location results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Longitude distribution
    axes[0, 0].hist(summary.X[:, 0], bins=30, alpha=0.7)
    axes[0, 0].set_xlabel('Longitude (°)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Longitude Distribution')
    
    # Latitude distribution
    axes[0, 1].hist(summary.X[:, 1], bins=30, alpha=0.7)
    axes[0, 1].set_xlabel('Latitude (°)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Latitude Distribution')
    
    # Depth distribution
    axes[1, 0].hist(summary.X[:, 2], bins=30, alpha=0.7)
    axes[1, 0].set_xlabel('Depth (km)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Depth Distribution')
    
    # Map view
    axes[1, 1].scatter(summary.X[:, 0], summary.X[:, 1], c=summary.X[:, 2], 
                      cmap='viridis', alpha=0.6)
    axes[1, 1].set_xlabel('Longitude (°)')
    axes[1, 1].set_ylabel('Latitude (°)')
    axes[1, 1].set_title('Event Locations')
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Depth (km)')
    
    plt.tight_layout()
    plt.savefig('location_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    summary = analyze_results()
    plot_results(summary)
```

### Memory-Efficient Analysis

**File: `memory_efficient_analysis.py`**

```python
#!/usr/bin/env python3
"""
Memory-efficient analysis using MCMC thinning.
"""

import json
import time
import psutil
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz

def memory_efficient_analysis():
    # Load parameters
    with open('params.json', 'r') as f:
        params = json.load(f)
    
    # Test different thinning factors
    thinning_factors = [1, 2, 5, 10, 20]
    
    for thin in thinning_factors:
        print(f"\nTesting thinning factor: {thin}")
        
        # Monitor memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Read samples with thinning
        samples = read_all_samples(params, thin=thin)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        read_time = time.time() - start_time
        
        if samples:
            n_samples = samples['longitude'].shape[1]
            print(f"  Samples per event: {n_samples}")
            print(f"  Memory usage: {memory_after:.1f} MB (change: {memory_after - memory_before:+.1f} MB)")
            print(f"  Read time: {read_time:.2f} seconds")
            
            # Compute summary
            start_time = time.time()
            summary = compute_cat_dd_and_xyz(samples, burn_in=20, thin=1)
            analysis_time = time.time() - start_time
            
            print(f"  Analysis time: {analysis_time:.2f} seconds")
            print(f"  Total time: {read_time + analysis_time:.2f} seconds")
        else:
            print("  No samples found")

if __name__ == "__main__":
    memory_efficient_analysis()
```

## Performance Optimization

### Multi-GPU Configuration

**Configuration: `params_multi_gpu.json`**

```json
{
    "dtime_file": "data/dtimes.csv",
    "catalog_infile": "data/events.csv",
    "catalog_outfile": "results/SPIDER_out.cat",
    "samples_outfile": "results/SPIDER_samples.h5",
    "model_file": "models/eikonet_model.pt",
    
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
    
    "devices": [0, 1, 2, 3],
    
    "num_workers": 4,
    
    "use_wandb": true,
    "wandb_project_name": "spider-multi-gpu"
}
```

### Large Dataset Handling

**File: `large_dataset_example.py`**

```python
#!/usr/bin/env python3
"""
Example for handling large datasets efficiently.
"""

import json
import torch
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz

def handle_large_dataset():
    # Load parameters
    with open('params_large.json', 'r') as f:
        params = json.load(f)
    
    # Strategy 1: Heavy thinning for exploration
    print("Strategy 1: Heavy thinning for exploration")
    samples_explore = read_all_samples(params, thin=20)
    
    if samples_explore:
        summary_explore = compute_cat_dd_and_xyz(samples_explore, burn_in=10, thin=1)
        print(f"Exploration: {summary_explore.X.shape[0]} events located")
    
    # Strategy 2: Moderate thinning for analysis
    print("\nStrategy 2: Moderate thinning for analysis")
    samples_analysis = read_all_samples(params, thin=5)
    
    if samples_analysis:
        summary_analysis = compute_cat_dd_and_xyz(samples_analysis, burn_in=50, thin=1)
        print(f"Analysis: {summary_analysis.X.shape[0]} events located")
    
    # Strategy 3: Light thinning for final results
    print("\nStrategy 3: Light thinning for final results")
    samples_final = read_all_samples(params, thin=2)
    
    if samples_final:
        summary_final = compute_cat_dd_and_xyz(samples_final, burn_in=100, thin=1)
        print(f"Final: {summary_final.X.shape[0]} events located")
        
        # Save final results
        summary_final.save_catalog("final_locations_large.csv")

def gpu_memory_optimization():
    """Example of GPU memory optimization."""
    
    # Check available GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {memory:.1f} GB")
    
    # Adjust batch sizes based on GPU memory
    gpu_memory_gb = 8  # Adjust based on your GPU
    
    if gpu_memory_gb < 4:
        batch_size = 5000
    elif gpu_memory_gb < 8:
        batch_size = 10000
    else:
        batch_size = 20000
    
    print(f"Recommended batch size: {batch_size}")
    
    return batch_size

if __name__ == "__main__":
    handle_large_dataset()
    gpu_memory_optimization()
```

## Real-World Scenarios

### Ridgecrest Earthquake Sequence

**File: `ridgecrest_example.py`**

```python
#!/usr/bin/env python3
"""
Example using Ridgecrest earthquake sequence data.
"""

import json
import numpy as np
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz

def ridgecrest_analysis():
    # Load Ridgecrest-specific parameters
    with open('ridgecrest_params.json', 'r') as f:
        params = json.load(f)
    
    # Read samples
    samples = read_all_samples(params, thin=5)
    
    if not samples:
        print("No samples found. Run SPIDER first.")
        return
    
    # Compute summary
    summary = compute_cat_dd_and_xyz(samples, burn_in=100, thin=1)
    
    # Ridgecrest-specific analysis
    print("Ridgecrest Earthquake Sequence Analysis")
    print("=" * 50)
    
    # Filter events by time (July 2019)
    july_events = summary.X[:, 3] >= np.datetime64('2019-07-01')
    july_events &= summary.X[:, 3] < np.datetime64('2019-08-01')
    
    print(f"Total events: {summary.X.shape[0]}")
    print(f"July 2019 events: {july_events.sum()}")
    
    # Spatial analysis
    lon_range = summary.X[:, 0].max() - summary.X[:, 0].min()
    lat_range = summary.X[:, 1].max() - summary.X[:, 1].min()
    depth_range = summary.X[:, 2].max() - summary.X[:, 2].min()
    
    print(f"Spatial extent:")
    print(f"  Longitude: {lon_range:.3f}°")
    print(f"  Latitude: {lat_range:.3f}°")
    print(f"  Depth: {depth_range:.1f} km")
    
    # Save results
    summary.save_catalog("ridgecrest_locations.csv")
    
    return summary

if __name__ == "__main__":
    ridgecrest_analysis()
```

### Synthetic Data Validation

**File: `synthetic_validation.py`**

```python
#!/usr/bin/env python3
"""
Example using synthetic data for validation.
"""

import json
import numpy as np
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz

def synthetic_validation():
    # Load synthetic data parameters
    with open('synthetic_params.json', 'r') as f:
        params = json.load(f)
    
    # Read true locations (if available)
    try:
        true_locations = np.loadtxt('synthetic_true_locations.txt')
        has_true = True
    except FileNotFoundError:
        has_true = False
        print("True locations not found. Skipping validation.")
    
    # Read SPIDER results
    samples = read_all_samples(params, thin=5)
    
    if not samples:
        print("No samples found. Run SPIDER first.")
        return
    
    # Compute summary
    summary = compute_cat_dd_and_xyz(samples, burn_in=100, thin=1)
    
    print("Synthetic Data Validation")
    print("=" * 40)
    
    if has_true:
        # Compare with true locations
        errors = summary.X[:, :3] - true_locations[:, :3]  # lon, lat, depth
        
        print(f"Location Errors:")
        print(f"  Mean longitude error: {np.abs(errors[:, 0]).mean():.4f}°")
        print(f"  Mean latitude error: {np.abs(errors[:, 1]).mean():.4f}°")
        print(f"  Mean depth error: {np.abs(errors[:, 2]).mean():.2f} km")
        
        print(f"  RMS longitude error: {np.sqrt((errors[:, 0]**2).mean()):.4f}°")
        print(f"  RMS latitude error: {np.sqrt((errors[:, 1]**2).mean()):.4f}°")
        print(f"  RMS depth error: {np.sqrt((errors[:, 2]**2).mean()):.2f} km")
    
    # Uncertainty analysis
    uncertainties = []
    for i in range(summary.X.shape[0]):
        unc = summary.get_uncertainty(i)
        uncertainties.append([
            unc['longitude_std'],
            unc['latitude_std'],
            unc['depth_std']
        ])
    
    uncertainties = np.array(uncertainties)
    
    print(f"\nUncertainty Statistics:")
    print(f"  Mean longitude uncertainty: {uncertainties[:, 0].mean():.4f}°")
    print(f"  Mean latitude uncertainty: {uncertainties[:, 1].mean():.4f}°")
    print(f"  Mean depth uncertainty: {uncertainties[:, 2].mean():.2f} km")
    
    # Save results
    summary.save_catalog("synthetic_results.csv")
    
    return summary

if __name__ == "__main__":
    synthetic_validation()
```

## Troubleshooting Examples

### GPU Memory Issues

**File: `gpu_memory_fix.py`**

```python
#!/usr/bin/env python3
"""
Example of fixing GPU memory issues.
"""

import json
import torch
from spider.io.samples import read_all_samples

def fix_gpu_memory_issues():
    # Load parameters
    with open('params.json', 'r') as f:
        params = json.load(f)
    
    print("GPU Memory Optimization Strategies")
    print("=" * 40)
    
    # Strategy 1: Reduce batch size
    print("\nStrategy 1: Reduce batch size")
    params_fixed = params.copy()
    params_fixed["batch_size_warmup"] = 5000
    params_fixed["batch_size_sgld"] = 5000
    
    print(f"  Reduced batch size to: {params_fixed['batch_size_warmup']}")
    
    # Strategy 2: Use heavy thinning
    print("\nStrategy 2: Use heavy thinning")
    try:
        samples_thinned = read_all_samples(params, thin=10)
        if samples_thinned:
            print(f"  Successfully read {samples_thinned['longitude'].shape[1]} samples per event")
    except RuntimeError as e:
        print(f"  Error: {e}")
    
    # Strategy 3: Use CPU backend
    print("\nStrategy 3: Use CPU backend")
    try:
        samples_cpu = read_all_samples(params, thin=5, backend='pandas')
        if samples_cpu:
            print(f"  Successfully read {samples_cpu['longitude'].shape[1]} samples per event")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Strategy 4: Check GPU memory
    if torch.cuda.is_available():
        print("\nStrategy 4: GPU memory status")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {cached:.2f} GB cached, {total:.1f} GB total")
            
            # Clear cache if needed
            if cached > total * 0.8:
                torch.cuda.empty_cache()
                print(f"  Cleared GPU {i} cache")

def create_memory_efficient_params():
    """Create memory-efficient parameter file."""
    
    params = {
        "dtime_file": "data/dtimes.csv",
        "catalog_infile": "data/events.csv",
        "catalog_outfile": "results/SPIDER_out.cat",
        "samples_outfile": "results/SPIDER_samples.h5",
        "model_file": "models/eikonet_model.pt",
        
        "lon_min": -120.0,
        "lat_min": 35.0,
        "z_min": -5.0,
        "z_max": 50.0,
        "scale": 200.0,
        
        "phase1_epochs": 500,  # Reduced
        "phase2_epochs": 100,  # Reduced
        "phase3_epochs": 250,  # Reduced
        "phase4_epochs": 5000, # Reduced
        
        "lr_warmup": 2e-3,
        "lr_sgld": 2e-3,
        "batch_size_warmup": 5000,  # Reduced
        "batch_size_sgld": 5000,    # Reduced
        
        "devices": [0],
        
        "save_every_n": 200,  # Increased
        
        "checkpoint_dir": "checkpoints/",
        "checkpoint_interval": 50,
        "min_samples_to_save": 5
    }
    
    with open('params_memory_efficient.json', 'w') as f:
        json.dump(params, f, indent=4)
    
    print("Created memory-efficient parameter file: params_memory_efficient.json")

if __name__ == "__main__":
    fix_gpu_memory_issues()
    create_memory_efficient_params()
```

### Data Format Issues

**File: `data_format_fix.py`**

```python
#!/usr/bin/env python3
"""
Example of fixing data format issues.
"""

import pandas as pd
import numpy as np

def fix_event_catalog():
    """Fix common issues in event catalog."""
    
    # Read problematic file
    try:
        df = pd.read_csv('problematic_events.csv')
    except FileNotFoundError:
        print("File not found. Creating example.")
        return
    
    print("Fixing event catalog...")
    
    # Fix 1: Ensure required columns
    required_cols = ['event_id', 'longitude', 'latitude', 'depth', 'origin_time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        # Add missing columns with defaults
        for col in missing_cols:
            if col == 'event_id':
                df[col] = [f'evt_{i:03d}' for i in range(len(df))]
            elif col in ['longitude', 'latitude', 'depth']:
                df[col] = 0.0
            elif col == 'origin_time':
                df[col] = '2023-01-01T00:00:00'
    
    # Fix 2: Ensure proper data types
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    
    # Fix 3: Handle missing values
    df = df.dropna(subset=['longitude', 'latitude', 'depth'])
    
    # Fix 4: Ensure proper time format
    df['origin_time'] = pd.to_datetime(df['origin_time'], errors='coerce')
    df = df.dropna(subset=['origin_time'])
    
    # Save fixed file
    df.to_csv('fixed_events.csv', index=False)
    print(f"Fixed event catalog saved: {len(df)} events")

def fix_differential_times():
    """Fix common issues in differential times."""
    
    try:
        df = pd.read_csv('problematic_dtimes.csv')
    except FileNotFoundError:
        print("File not found. Creating example.")
        return
    
    print("Fixing differential times...")
    
    # Fix 1: Ensure required columns
    required_cols = ['event1_id', 'event2_id', 'station', 'phase', 'dt_obs', 'dt_err']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        for col in missing_cols:
            if col in ['event1_id', 'event2_id', 'station']:
                df[col] = 'unknown'
            elif col in ['phase']:
                df[col] = 'P'
            elif col in ['dt_obs', 'dt_err']:
                df[col] = 0.0
    
    # Fix 2: Ensure proper data types
    df['dt_obs'] = pd.to_numeric(df['dt_obs'], errors='coerce')
    df['dt_err'] = pd.to_numeric(df['dt_err'], errors='coerce')
    
    # Fix 3: Handle missing values
    df = df.dropna(subset=['dt_obs', 'dt_err'])
    
    # Fix 4: Ensure positive errors
    df['dt_err'] = df['dt_err'].abs()
    
    # Fix 5: Filter valid phases
    valid_phases = ['P', 'S', 'p', 's']
    df = df[df['phase'].isin(valid_phases)]
    
    # Save fixed file
    df.to_csv('fixed_dtimes.csv', index=False)
    print(f"Fixed differential times saved: {len(df)} observations")

if __name__ == "__main__":
    fix_event_catalog()
    fix_differential_times()
```

---

For more examples, see the [example scripts](example_*.py) and [notebooks](notebooks/) in the repository.
