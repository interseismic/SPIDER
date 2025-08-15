# EikoNet Training Guide

This guide explains how to train EikoNet models for use with SPIDER earthquake location.

## Overview

EikoNet is a neural network that predicts travel times for seismic waves through a 3D velocity model. SPIDER requires a pre-trained EikoNet model to perform fast travel time calculations during earthquake location.

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- EikoNet training script (`eikonet_train.py`)
- Regional velocity model

## Training Configuration

### EikoNet JSON Parameters

Create an `eikonet.json` configuration file:

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

### Parameter Descriptions

| Parameter | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `velmod_file` | string | Path to velocity model CSV | Required | `"velmod.csv"` |
| `lon_min`, `lat_min` | float | Spatial domain bounds | Required | `-117.5, 33.0` |
| `z_min`, `z_max` | float | Depth bounds (km) | Required | `-5.0, 80.0` |
| `scale` | float | Spatial scale factor | Required | `400.0` |
| `model_file` | string | Output model path | Required | `"model.pt"` |
| `train_batch_size` | int | Training batch size | `512` | `512` |
| `val_batch_size` | int | Validation batch size | `10000` | `10000` |
| `n_train` | int | Training samples | `1000000` | `1000000` |
| `n_test` | int | Test samples | `2000000` | `2000000` |
| `n_epochs` | int | Training epochs | `1000` | `1000` |
| `lr` | float | Learning rate | `1e-3` | `1e-3` |

## Velocity Model Format

### CSV Structure

Create a CSV file with depth-dependent velocities:

```csv
depth,vs,vp
-5.0,3.3,6.0
0.0,3.3,6.0
5.0,3.4,6.1
10.0,3.5,6.2
15.0,3.6,6.3
20.0,3.7,6.4
25.0,3.8,6.5
30.0,3.9,6.6
35.0,4.0,6.7
40.0,4.1,6.8
45.0,4.2,6.9
50.0,4.3,7.0
55.0,4.4,7.1
60.0,4.5,7.2
65.0,4.6,7.3
70.0,4.7,7.4
75.0,4.8,7.5
80.0,4.9,7.6
```

### Column Descriptions

- **`depth`**: Depth in kilometers (negative for above surface)
- **`vs`**: S-wave velocity in km/s
- **`vp`**: P-wave velocity in km/s

### Velocity Model Guidelines

1. **Depth coverage**: Ensure coverage from `z_min` to `z_max`
2. **Smooth transitions**: Avoid sharp velocity jumps
3. **Regional accuracy**: Use region-specific velocity structure
4. **Resolution**: Include enough depth points for smooth interpolation

## Training Process

### Step 1: Prepare Velocity Model

```bash
# Create your velocity model CSV
# Example for Southern California
cat > velmod.csv << EOF
depth,vs,vp
-5.0,3.3,6.0
0.0,3.3,6.0
5.0,3.4,6.1
10.0,3.5,6.2
15.0,3.6,6.3
20.0,3.7,6.4
25.0,3.8,6.5
30.0,3.9,6.6
35.0,4.0,6.7
40.0,4.1,6.8
45.0,4.2,6.9
50.0,4.3,7.0
55.0,4.4,7.1
60.0,4.5,7.2
65.0,4.6,7.3
70.0,4.7,7.4
75.0,4.8,7.5
80.0,4.9,7.6
EOF
```

### Step 2: Configure Training

```bash
# Create eikonet.json configuration
cat > eikonet.json << EOF
{
    "velmod_file": "velmod.csv",
    "lon_min": -117.5,
    "lat_min": 33.0,
    "z_min": -5.0,
    "z_max": 80.0,
    "scale": 400.0,
    "model_file": "model.pt",
    "train_batch_size": 512,
    "val_batch_size": 10000,
    "n_train": 1000000,
    "n_test": 2000000,
    "n_epochs": 1000,
    "lr": 1e-3
}
EOF
```

### Step 3: Train the Model

```bash
# Run training
python eikonet_train.py eikonet.json
```

### Step 4: Verify Training

```bash
# Check that model was created
ls -la model.pt

# Test model loading
python -c "
import torch
model = torch.load('model.pt', map_location='cpu')
print('Model loaded successfully!')
print(f'Model type: {type(model)}')
"
```

## Training Examples

### Example 1: Cahuilla Dataset

**Configuration** (`cahuilla/eikonet.json`):
```json
{
    "velmod_file": "/home/zross/git/SPIDER/cahuilla/velmod.csv",
    "lon_min": -117.5,
    "lat_min": 33.0,
    "z_min": -5.0,
    "z_max": 80.0,
    "scale": 400.0,
    "model_file": "/home/zross/git/SPIDER/cahuilla/model.pt",
    "train_batch_size": 512,
    "val_batch_size": 10000,
    "n_train": 1000000,
    "n_test": 2000000,
    "n_epochs": 1000,
    "lr": 1e-3
}
```

**Training command**:
```bash
cd cahuilla
python eikonet_train.py eikonet.json
```

### Example 2: Ridgecrest Dataset

**Configuration** (`ridgecrest/eikonet.json`):
```json
{
    "velmod_file": "ridgecrest/velmod.csv",
    "lon_min": -118.0,
    "lat_min": 35.5,
    "z_min": -5.0,
    "z_max": 50.0,
    "scale": 300.0,
    "model_file": "ridgecrest/model.pt",
    "train_batch_size": 256,
    "val_batch_size": 5000,
    "n_train": 500000,
    "n_test": 1000000,
    "n_epochs": 500,
    "lr": 5e-4
}
```

## Training Tips

### Performance Optimization

1. **GPU Memory**: Adjust batch sizes based on available GPU memory
   ```json
   {
       "train_batch_size": 256,  // Reduce if out of memory
       "val_batch_size": 5000
   }
   ```

2. **Training Speed**: Use more samples for better accuracy
   ```json
   {
       "n_train": 2000000,  // More training samples
       "n_test": 4000000    // More test samples
   }
   ```

3. **Learning Rate**: Start with 1e-3, reduce if training is unstable
   ```json
   {
       "lr": 5e-4  // Reduce if loss doesn't converge
   }
   ```

### Domain Configuration

1. **Spatial Bounds**: Ensure coverage of your study area
   ```json
   {
       "lon_min": -120.0,  // Cover your earthquake locations
       "lat_min": 35.0,
       "lon_max": -115.0,  // Add if needed
       "lat_max": 40.0     // Add if needed
   }
   ```

2. **Depth Range**: Include all relevant earthquake depths
   ```json
   {
       "z_min": -5.0,   // Surface to maximum depth
       "z_max": 50.0    // Adjust based on your data
   }
   ```

3. **Scale Factor**: Adjust based on domain size
   ```json
   {
       "scale": 200.0   // Smaller for smaller domains
   }
   ```

### Quality Assurance

1. **Check Training Loss**: Monitor convergence during training
2. **Validate Model**: Test with known source-receiver pairs
3. **Compare with Ray Tracing**: Verify against traditional methods
4. **Check File Size**: Typical model size is 500KB-2MB

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
```json
{
    "train_batch_size": 128,  // Reduce batch size
    "val_batch_size": 1000
}
```

**"Training loss not converging"**
```json
{
    "lr": 5e-4,  // Reduce learning rate
    "n_epochs": 2000  // Increase epochs
}
```

**"Model file not found"**
```bash
# Check file paths in eikonet.json
# Ensure velmod.csv exists
ls -la velmod.csv
```

**"Invalid velocity model"**
```bash
# Check CSV format
head -5 velmod.csv
# Ensure depth,vs,vp columns exist
```

### Training Monitoring

Monitor training progress:
- Loss should decrease over time
- Validation accuracy should improve
- Training time depends on dataset size and hardware

## Integration with SPIDER

After training, use the model in SPIDER:

```json
{
    "model_file": "path/to/trained/model.pt",
    "scale": 400.0,  // Must match training scale
    "lon_min": -117.5,  // Must match training bounds
    "lat_min": 33.0
}
```

## Advanced Configuration

### Custom Training Parameters

```json
{
    "velmod_file": "velmod.csv",
    "lon_min": -117.5,
    "lat_min": 33.0,
    "z_min": -5.0,
    "z_max": 80.0,
    "scale": 400.0,
    "model_file": "model.pt",
    "train_batch_size": 512,
    "val_batch_size": 10000,
    "n_train": 1000000,
    "n_test": 2000000,
    "n_epochs": 1000,
    "lr": 1e-3,
    "lr_scheduler": "cosine",
    "weight_decay": 1e-4,
    "gradient_clip": 1.0
}
```

### Multi-GPU Training

```json
{
    "devices": [0, 1, 2, 3],
    "train_batch_size": 2048,
    "val_batch_size": 40000
}
```

---

For more information, see the [main README](README.md) and [API reference](API_REFERENCE.md).
