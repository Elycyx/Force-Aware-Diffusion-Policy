# Force-Aware Diffusion Policy (FADP)

<p align="center">
  <img src="media/teaser.png" alt="FADP Overview" width="100%"/>
</p>

**Force-Aware Diffusion Policy (FADP)** is an enhanced version of [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) that incorporates force/torque sensor data for robust robot manipulation. This implementation focuses on real robot deployment with multi-modal sensory inputs.

## 🌟 Key Features

- **Force/Torque Integration**: Incorporates 6-DOF force/torque measurements (fx, fy, fz, mx, my, mz) alongside visual observations
- **7-DOF Action Space**: Outputs delta end-effector pose (dx, dy, dz, drx, dry, drz) plus gripper command
- **Modular Design**: Clean separation between vision and force encoders for easy customization
- **Real Robot Ready**: Streamlined codebase focused on real-world deployment
- **UNet-based Diffusion**: Leverages conditional diffusion models for multi-modal policy learning

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Architecture](#architecture)
- [Customization Guide](#customization-guide)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🛠️ Installation

### Prerequisites

- Ubuntu 20.04 or later
- CUDA 11.3+ (for GPU training)
- Python 3.9+
- Conda or Miniconda

### Method 1: Using Conda (Recommended)

This method installs all dependencies including simulation environments:

```bash
# Clone the repository
git clone https://github.com/Elycyx/Force-Aware-Diffusion-Policy.git
cd Force-Aware-Diffusion-Policy

# Create conda environment from yaml
conda create -n fadp python=3.10

# Activate environment
conda activate fadp

# Install FADP package in editable mode
pip install -e .
```


## 🚀 Quick Start

### 1. Test the Model with Dummy Data

```bash
python test_fadp_model.py
```

This will:
- Initialize the ForceEncoder (6D → feature_dim)
- Create a complete FADP policy
- Test forward/backward passes
- Print model architecture and parameter statistics

### 2. Train on Your Data

```bash
python train.py --config-name=train_force_aware_diffusion_policy_real
```

### 3. Evaluate on Real Robot

```bash
python eval_real_robot.py \
    -i data/outputs/your_checkpoint.ckpt \
    -o eval_results \
    --robot_ip 192.168.1.100
```

## 📊 Data Preparation

> **📘 Documentation:**
> - [DATASET_FORMAT.md](DATASET_FORMAT.md) - Detailed dataset format specification
> - [FADP_DATASET.md](FADP_DATASET.md) - **NEW**: FADP Dataset class with relative action support
> - [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Comprehensive training guide

### Data Format Overview

FADP supports two dataset classes:

1. **FADPDataset (Recommended)**: Modern, efficient dataset with relative action support
2. **RealPushTImageDataset**: Legacy dataset for backward compatibility

FADP expects data in **zarr** format with the following structure:

```
your_dataset/
├── data.zarr/
│   ├── camera_0/        # RGB images: (T, H, W, 3)
│   ├── camera_1/        # Optional additional camera
│   ├── force/           # Force/torque data: (T, 6)
│   └── action/          # Actions: (T, 7)
└── meta/
    └── episode_ends.json  # Episode boundaries
```

### Force Data Format

Force data should be a `(T, 6)` float32 array where:
- `force[:, 0:3]` = Forces (fx, fy, fz) in Newtons
- `force[:, 3:6]` = Torques (mx, my, mz) in Newton-meters

### Convert Existing Data

If you have data in HDF5 format, use our conversion script:

```bash
# Convert HDF5 episodes to FADP format
python scripts/convert_hdf5_to_fadp.py \
    --input "data/session_*" \
    --output data/fadp_dataset \
    --image-size 240 320

# See scripts/README.md for more options
```

### Example: Creating a Dataset Manually

```python
import zarr
import numpy as np

# Create zarr store
store = zarr.DirectoryStore('my_dataset/data.zarr')
root = zarr.group(store=store, overwrite=True)

# Save data
root.create_dataset('camera_0', data=rgb_images, dtype='uint8')
root.create_dataset('force', data=force_readings, dtype='float32')
root.create_dataset('action', data=actions, dtype='float32')

# Save episode boundaries
import json
episode_ends = [100, 250, 400]  # End indices of each episode
with open('my_dataset/meta/episode_ends.json', 'w') as f:
    json.dump(episode_ends, f)
```

## 🎯 Training

> **📘 For complete training guide with troubleshooting, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)**

### Basic Training

#### Using FADPDataset (Recommended - with relative actions)

```bash
python train.py --config-name=train_fadp
```

This uses the new `FADPDataset` which supports relative actions (delta from first state).

#### Using RealPushTImageDataset (Legacy - absolute actions)

```bash
python train.py --config-name=train_force_aware_diffusion_policy_real
```

### Custom Configuration

Create a custom config file (e.g., `my_config.yaml`):

```yaml
defaults:
  - _self_
  - task: real_pusht_image

name: my_experiment
_target_: fadp.workspace.train_force_aware_diffusion_policy_workspace.TrainForceAwareDiffusionPolicyWorkspace

# Model parameters
policy:
  _target_: fadp.policy.force_aware_diffusion_policy.ForceAwareDiffusionPolicy
  horizon: 16
  n_obs_steps: 2
  n_action_steps: 8
  force_encoder_hidden_dim: 512  # Adjust force encoder size

# Training parameters
training:
  num_epochs: 600
  batch_size: 64
  lr: 1.0e-4
```

Then train:

```bash
python train.py --config-name=my_config
```

### Monitor Training

Training logs are saved to Weights & Biases. View them at:
```
https://wandb.ai/your-project/fadp_experiments
```

## 🤖 Evaluation

### Real Robot Evaluation

```bash
python eval_real_robot.py \
    -i checkpoints/best_model.ckpt \
    -o eval_results \
    --robot_ip 192.168.1.100 \
    --frequency 10
```

### Key Arguments

- `-i, --input`: Path to checkpoint file
- `-o, --output`: Directory to save evaluation results
- `--robot_ip`: IP address of your robot
- `--frequency`: Control frequency in Hz (default: 10)
- `--steps_per_inference`: Action horizon (default: 6)

## 🏗️ Architecture

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                     FADP Architecture                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Inputs:                                                  │
│  ┌──────────────┐        ┌──────────────┐               │
│  │ RGB Image    │        │ Force/Torque │               │
│  │ (H, W, 3)    │        │ (6,)         │               │
│  └──────┬───────┘        └──────┬───────┘               │
│         │                       │                        │
│         ▼                       ▼                        │
│  ┌──────────────┐        ┌──────────────┐               │
│  │ ResNet-18    │        │ Force MLP    │               │
│  │ Encoder      │        │ Encoder      │               │
│  └──────┬───────┘        └──────┬───────┘               │
│         │                       │                        │
│         │  RGB Features (64)    │  Force Features (64)  │
│         └───────────┬───────────┘                        │
│                     │                                    │
│                     ▼                                    │
│         ┌───────────────────────┐                        │
│         │  Concatenate Features │                        │
│         │  Global Condition     │                        │
│         └──────────┬────────────┘                        │
│                    │                                     │
│                    ▼                                     │
│         ┌─────────────────────┐                          │
│         │ Conditional UNet 1D │                          │
│         │ (Diffusion Model)   │                          │
│         └──────────┬──────────┘                          │
│                    │                                     │
│                    ▼                                     │
│  Output: Action Sequence (n_steps, 7)                    │
│          [dx, dy, dz, drx, dry, drz, gripper]            │
└─────────────────────────────────────────────────────────┘
```

### Components

1. **Vision Encoder**: ResNet-18 pre-trained on ImageNet
2. **Force Encoder**: 2-layer MLP with Swish activation
   ```
   Input(6) → Linear(6→512) → Swish → Linear(512→64) → Output(64)
   ```
3. **Diffusion Model**: Conditional UNet 1D with FiLM conditioning
4. **Noise Scheduler**: DDIM with 100 training steps, 16 inference steps

### Model Statistics

```
Component                     Parameters    Size (MB)
─────────────────────────────────────────────────────
RGB Encoder (ResNet-18)       11.2M         42.7
Force Encoder (MLP)           0.04M         0.15
Diffusion UNet                66.7M         254.5
─────────────────────────────────────────────────────
Total                         78.0M         297.4
```

## 🔧 Customization Guide

### 1. Modify Force Encoder Architecture

Edit `fadp/model/common/force_encoder.py`:

```python
class ForceEncoder(nn.Module):
    def __init__(self, input_dim=6, output_dim=256, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Add more layers
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
```

### 2. Change Action Dimensions

To use a different action space, modify `fadp/config/task/real_pusht_image.yaml`:

```yaml
shape_meta:
  action:
    shape: [10]  # Change to your action dimension
```

Then update your robot interface accordingly.

### 3. Add Additional Sensors

To add tactile or other sensors:

1. **Update shape_meta**:
```yaml
shape_meta:
  obs:
    camera_0: {shape: [3, 240, 320], type: rgb}
    force: {shape: [6], type: low_dim}
    tactile: {shape: [16], type: low_dim}  # New sensor
```

2. **Create encoder** (if needed):
```python
# In ForceAwareDiffusionPolicy.__init__
if 'tactile' in obs_shape_meta:
    self.tactile_encoder = TactileEncoder(...)
```

3. **Modify forward pass**:
```python
# In compute_loss and predict_action
tactile_features = self.tactile_encoder(tactile_data)
all_features = torch.cat([rgb_features, force_features, tactile_features], dim=-1)
```

### 4. Use Single Camera Instead of Multiple

Modify `fadp/config/task/real_pusht_image.yaml`:

```yaml
shape_meta:
  obs:
    image:  # Use 'image' instead of 'camera_0', 'camera_1'
      shape: [3, 240, 320]
      type: rgb
    force:
      shape: [6]
      type: low_dim
```

### 5. Adjust Training Hyperparameters

Common hyperparameters to tune:

```yaml
# In training config
horizon: 16                    # Prediction horizon
n_obs_steps: 2                 # Number of observation frames
n_action_steps: 8              # Action execution steps
force_encoder_hidden_dim: 512  # Force encoder size

training:
  batch_size: 64               # Reduce if OOM
  lr: 1.0e-4                   # Learning rate
  num_epochs: 600              # Training epochs
```

### 6. Implement Custom Robot Interface

Edit `fadp/real_world/real_env.py` to add force sensor support:

```python
class RealEnv:
    def __init__(self, force_sensor_config=None, **kwargs):
        # Initialize existing components
        self.realsense = ...
        self.robot = ...
        
        # Add force sensor
        if force_sensor_config is not None:
            self.force_sensor = self._init_force_sensor(force_sensor_config)
    
    def _init_force_sensor(self, config):
        """Initialize your force sensor hardware"""
        # Example for ATI sensor
        from your_sensor_driver import ForceTorqueSensor
        return ForceTorqueSensor(
            ip=config['ip'],
            port=config['port']
        )
    
    def get_obs(self):
        """Get observations including force data"""
        obs = super().get_obs()
        
        # Read force sensor
        if self.force_sensor is not None:
            force_reading = self.force_sensor.read()
            obs['force'] = force_reading  # Shape: (6,)
        
        return obs
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
dataloader:
  batch_size: 32  # Reduce from 64
```

**2. Import Errors**
```bash
# Reinstall the package
pip install -e .
```

**3. Force Data Shape Mismatch**
Ensure your force data is `(T, 6)` with columns: `[fx, fy, fz, mx, my, mz]`

**4. Checkpoint Loading Fails**
Check that the model architecture in your config matches the checkpoint:
```python
# In eval script, verify:
print(f"Config action dim: {cfg.task.shape_meta['action']['shape']}")
```

### Debug Mode

Enable debug mode for detailed logging:

```yaml
training:
  debug: True
  max_train_steps: 3
  max_val_steps: 3
```

## 📝 File Structure

```
force-aware-diffusion-policy/
├── fadp/                          # Main package
│   ├── policy/
│   │   ├── base_image_policy.py
│   │   └── force_aware_diffusion_policy.py  # Main policy
│   ├── model/
│   │   ├── common/
│   │   │   └── force_encoder.py   # Force encoder
│   │   └── diffusion/
│   │       └── conditional_unet1d.py
│   ├── dataset/
│   │   └── real_pusht_image_dataset.py
│   ├── workspace/
│   │   └── train_force_aware_diffusion_policy_workspace.py
│   ├── real_world/               # Real robot utilities
│   │   ├── real_env.py
│   │   ├── real_inference_util.py
│   │   └── ...
│   └── config/                   # Configuration files
│       ├── task/
│       │   └── real_pusht_image.yaml
│       └── train_force_aware_diffusion_policy_real.yaml
├── train.py                      # Training script
├── eval_real_robot.py           # Evaluation script
├── test_fadp_model.py           # Unit tests
└── README.md                    # This file
```
