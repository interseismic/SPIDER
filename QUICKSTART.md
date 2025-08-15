# SPIDER Quick Start Guide

Get up and running with SPIDER in minutes! This guide will help you install SPIDER and run your first earthquake location.

## 🚀 5-Minute Setup

### 1. Install SPIDER

```bash
# Clone the repository
git clone https://github.com/your-username/SPIDER.git
cd SPIDER

# Install in development mode
pip install -e .

# Verify installation
python -c "import spider; print('SPIDER installed successfully!')"
```

### 2. Train EikoNet Model

Before running SPIDER, you need a trained EikoNet model for travel time prediction.

**Create velocity model** (`velmod.csv`):
```csv
depth,vs,vp
-5.0,3.3,6.0
0.0,3.3,6.0
5.0,3.4,6.1
10.0,3.5,6.2
15.0,3.6,6.3
20.0,3.7,6.4
```

**Create EikoNet config** (`eikonet.json`):
```json
{
    "velmod_file": "velmod.csv",
    "lon_min": -120.0,
    "lat_min": 35.0,
    "z_min": -5.0,
    "z_max": 50.0,
    "scale": 200.0,
    "model_file": "model.pt",
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
python eikonet_train.py eikonet.json
```

### 3. Prepare Sample Data

Create these two CSV files:

**`events.csv`** (Event catalog):
```csv
event_id,longitude,latitude,depth,origin_time
evt_001,-117.5,34.2,10.5,2023-01-01T12:00:00
evt_002,-117.6,34.3,12.1,2023-01-01T12:05:00
evt_003,-117.4,34.1,8.9,2023-01-01T12:10:00
```

**`dtimes.csv`** (Differential times):
```csv
event1_id,event2_id,station,phase,dt_obs,dt_err
evt_001,evt_002,STA1,P,0.5,0.1
evt_001,evt_002,STA1,S,1.2,0.15
evt_001,evt_003,STA1,P,0.8,0.12
evt_001,evt_003,STA1,S,1.8,0.18
evt_002,evt_003,STA1,P,0.3,0.08
evt_002,evt_003,STA1,S,0.6,0.12
```

### 4. Create Configuration

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
    
    "phase1_epochs": 100,
    "phase2_epochs": 50,
    "phase3_epochs": 100,
    "phase4_epochs": 500,
    
    "lr_warmup": 2e-3,
    "lr_sgld": 2e-3,
    "batch_size_warmup": 1000,
    "batch_size_sgld": 1000,
    
    "devices": [0]
}
```

### 5. Run SPIDER

```bash
python SPIDER.py params.json
```

### 6. Analyze Results

```python
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz

# Load parameters
with open('params.json', 'r') as f:
    params = json.load(f)

# Read samples
samples = read_all_samples(params)

# Compute summary
summary = compute_cat_dd_and_xyz(samples, burn_in=50)

# View results
print(f"Located {summary.X.shape[0]} events")
print(f"Mean longitude: {summary.X[:, 0].mean():.4f}°")
print(f"Mean latitude: {summary.X[:, 1].mean():.4f}°")
print(f"Mean depth: {summary.X[:, 2].mean():.2f} km")
```

## 📋 What You Need

### Required Files

1. **EikoNet Model**: Pre-trained neural network model (`.pt` file) - **You must train this first**
2. **Velocity Model**: CSV with depth-dependent velocities (`velmod.csv`)
3. **Event Catalog**: CSV with event locations and times
4. **Differential Times**: CSV with phase arrival time differences
5. **Configuration**: JSON parameter file

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible (recommended)
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **Storage**: 10GB+ free space for samples

## 🔧 Common Issues & Solutions

### "Model file not found"
```bash
# Download a pre-trained EikoNet model or train your own
# Update the "model_file" path in params.json
```

### "CUDA out of memory"
```json
{
    "batch_size_warmup": 500,
    "batch_size_sgld": 500
}
```

### "No samples found"
```python
# Check that SPIDER completed successfully
# Verify the samples_outfile path in params.json
```

## 📚 Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Explore examples**: [docs/EXAMPLES.md](docs/EXAMPLES.md)
3. **Check API reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
4. **Try notebooks**: [notebooks/](notebooks/)

## 🆘 Need Help?

- **Documentation**: [README.md](README.md)
- **Examples**: [docs/EXAMPLES.md](docs/EXAMPLES.md)
- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions

---

**Ready to locate earthquakes? Start with the 5-minute setup above!**
