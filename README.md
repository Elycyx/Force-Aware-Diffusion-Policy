# Force-Aware Diffusion Policy


## Overview

This project implements a complete pipeline for learning manipulation policies from demonstration data:

1. **Data Conversion**: Convert HDF5 demonstration data to zarr format for efficient training
2. **Dataset Support**: Support for FADP Force (multimodal vision+force), basic FADP, and UMI datasets
3. **Training**: Train diffusion-based policies with advanced multimodal encoders (DINOv3 + Force)
4. **Evaluation**: Test trained models and visualize performance with detailed metrics
5. **Force Prediction**: Predict future force/torque alongside actions for contact-rich tasks


## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU support)
- Conda (recommended)

### Setup

```bash
git clone https://github.com/Elycyx/Force-Aware-Diffusion-Policy.git
cd Force-Aware-Diffusion-Policy

pip install -e .
```


## Quick Start

### 1. Convert Demonstration Data

Convert your HDF5 session data to zarr format:

```bash
# Basic conversion
python convert_session_to_zarr.py -i data/session_xxxx

```

**Command-line Options:**
- `-i, --input`: Input session directory path (required)
- `-o, --output`: Output zarr file path (default: `<input>/dataset.zarr.zip`)
- `-s, --image-size`: Target image size as WIDTHxHEIGHT (default: 224x224)
- `-n, --max-episodes`: Maximum number of episodes to convert (default: all)
- `-w, --num-workers`: Number of parallel threads/processes (default: 4)
- `--use-multiprocessing`: Enable multi-process parallel processing (faster but uses more memory)
- `--fast-save`: Use fast save mode (faster but larger files)

### 2. Inspect Converted Data

Verify your converted dataset:

```bash
# Basic inspection
python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip
```

### 3. Train a Policy

Train a diffusion policy using the converted data:

```bash
# Train with FADP Force (vision + force, RECOMMENDED)
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \

```

### 4. Test Trained Model

Evaluate your trained model:

```bash
# Test FADP Force model (13D output)
python test_fadp_model.py \
    --checkpoint data/outputs/<experiment>/checkpoints/latest.ckpt \
    --dataset-path data/session_20251025_142256/dataset.zarr.zip

# Test on training set
python test_fadp_model.py \
    --checkpoint data/outputs/<experiment>/checkpoints/latest.ckpt \
    --dataset-path data/session_20251025_142256/dataset.zarr.zip \
    --dataset-type train
```


## Data Format

### Input (HDF5)

Each episode file contains:
- `action`: (T, 7) - Action data [x, y, z, rx, ry, rz, gripper]
- `state`: (T, 7) - State data [x, y, z, rx, ry, rz, gripper]
- `image`: (T, 480, 640, 3) - RGB images
- `force`: (T, 6) - Force/torque sensor data [fx, fy, fz, mx, my, mz]
- `timestamp*`: Various timestamps

### Output (Zarr)

The converted zarr file contains:
- `data/camera0_rgb`: RGB images (T, H, W, 3)
- `data/robot0_eef_pos`: End-effector position (T, 3)
- `data/robot0_eef_rot_axis_angle`: End-effector rotation (T, 3)
- `data/robot0_gripper_width`: Gripper width (T, 1)
- `data/force`: Force/torque data (T, 6)
- `data/action`: Action data (T, 7)
- `meta/episode_ends`: Episode end indices

**Note**: The conversion script automatically:
- Resizes images to the target size (default: 224x224)
- Adds π/2 to the rz component of actions
- Handles missing force data with zero arrays
- Validates data consistency


## Configuration

### Data Augmentation (FADP)

Configure pose noise augmentation in the training config:

```yaml
# Single value - same noise for all 6 dimensions [x, y, z, rx, ry, rz]
pose_noise_scale: 0.05

# List - different noise for each dimension
pose_noise_scale: [0.01, 0.01, 0.01, 0.05, 0.05, 0.05]
```

**Command-line override**:
```bash
# Disable noise
python train.py --config-name=train_diffusion_unet_timm_fadp_workspace \
    task.dataset.pose_noise_scale=0.0

# Custom noise
python train.py --config-name=train_diffusion_unet_timm_fadp_workspace \
    task.dataset.pose_noise_scale=[0.01,0.01,0.01,0.05,0.05,0.05]
```

**Note**: Validation set never uses noise augmentation.

### FADP Force Configuration

The FADP Force model supports independent history windows and force loss weighting:

```yaml
# History windows
img_obs_horizon: 2      # Number of RGB history steps (default: 2)
force_obs_horizon: 6    # Number of force history steps (default: 6)
action_horizon: 16      # Number of future action steps (default: 16)

# Force loss weight
force_loss_weight: 1.0  # Weight for force prediction loss (default: 1.0)

# DINOv3 model selection
dinov3_model: facebook/dinov3-vit-base-pretrain-lvd1689m  # HuggingFace model name
```

**Key Parameters**:
- `img_obs_horizon`: Vision observations are averaged over time steps
- `force_obs_horizon`: Force observations are flattened and processed by MLP
- `force_loss_weight`: Balance between action loss and force loss (0.5-2.0 typical range)

**Configuration file**: `diffusion_policy/config/task/fadp_force.yaml`

## Project Structure

```
universal_manipulation_interface/
├── convert_session_to_zarr.py         # Data conversion script
├── train.py                            # Training script
├── test_fadp_model.py                 # Model testing script (7D/13D support)
├── inspect_zarr.py                     # Zarr inspection utility
├── inspect_fadp_dataset.py            # FADP dataset inspection
├── visualize_obs_future_force.py      # Force data visualization
├── diffusion_policy/                   # Core diffusion policy framework
│   ├── config/                         # Training configurations
│   │   ├── task/
│   │   │   ├── fadp.yaml              # Basic FADP config (7D action)
│   │   │   ├── fadp_force.yaml        # FADP Force config (13D action)
│   │   │   └── umi.yaml               # UMI config
│   │   ├── train_diffusion_unet_timm_fadp_workspace.yaml
│   │   ├── train_diffusion_unet_fadp_force_workspace.yaml
│   │   └── train_*.yaml
│   ├── dataset/                        # Dataset classes
│   │   ├── fadp_dataset.py            # FADP dataset (supports 7D/13D)
│   │   └── umi_dataset.py             # UMI dataset
│   ├── policy/                         # Policy implementations
│   │   ├── diffusion_unet_fadp_policy.py  # FADP Force policy
│   │   └── diffusion_unet_timm_policy.py  # Basic policy
│   ├── model/                          # Model architectures
│   │   └── vision/
│   │       ├── fadp_encoder.py        # DINOv3 + Force encoder
│   │       └── timm_obs_encoder.py    # Timm encoder
│   └── workspace/                      # Training workspaces
│       ├── train_diffusion_unet_fadp_workspace.py  # FADP workspace
│       └── train_diffusion_unet_image_workspace.py # Basic workspace
└── data/                               # Data directory
    └── session_*/                      # Session data directories
```

## Code Usage Examples

### Using FADP in Your Code

#### Loading a Trained Model

```python
import torch
from omegaconf import OmegaConf
from diffusion_policy.workspace.train_diffusion_unet_fadp_workspace import TrainDiffusionUnetFADPWorkspace

# Load checkpoint
checkpoint_path = "data/outputs/experiment/checkpoints/latest.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Get configuration
cfg = checkpoint['cfg']

# Create workspace
workspace = TrainDiffusionUnetFADPWorkspace(cfg)

# Load model weights
workspace.load_payload(checkpoint, exclude_keys=None, include_keys=None)

# Get policy
policy = workspace.model
policy.eval()
```

#### Making Predictions

```python
import torch

# Prepare observations
obs = {
    'camera0_rgb': torch.randn(1, 2, 3, 224, 224),  # (B, T_img, C, H, W)
    'force': torch.randn(1, 6, 6)                    # (B, T_force, force_dim)
}

# Predict action and force
with torch.no_grad():
    result = policy.predict_action(obs)

# Extract results
action_only = result['action_only']      # (B, T_pred, 7) - pose only
force_pred = result['force_pred']        # (B, T_pred, 6) - predicted force
action = result['action']                # (B, T_pred, 13) - combined output

print(f"Predicted action shape: {action_only.shape}")
print(f"Predicted force shape: {force_pred.shape}")
```

#### Understanding the Output

```python
# The 13D output is structured as:
# - Dimensions 0-6: Relative pose [x, y, z, rx, ry, rz, gripper]
# - Dimensions 7-12: Relative force [fx, fy, fz, mx, my, mz]

# Action components
position_delta = action[:, :, 0:3]      # Position change
rotation_delta = action[:, :, 3:6]      # Rotation change (axis-angle)
gripper_delta = action[:, :, 6:7]       # Gripper change

# Force components (relative to current force)
force_delta = action[:, :, 7:13]        # Force change
linear_force = force_delta[:, :, 0:3]   # fx, fy, fz
angular_force = force_delta[:, :, 3:6]  # mx, my, mz
```

#### Custom Dataset Configuration

```python
from omegaconf import OmegaConf
from diffusion_policy.dataset.fadp_dataset import FadpDataset

# Load configuration
cfg = OmegaConf.load('diffusion_policy/config/task/fadp_force.yaml')
OmegaConf.resolve(cfg)

# Create dataset with custom parameters
dataset = FadpDataset(
    shape_meta=OmegaConf.to_container(cfg.shape_meta, resolve=True),
    dataset_path='data/session_20251025_142256/dataset.zarr.zip',
    pose_repr=OmegaConf.to_container(cfg.get('pose_repr', {}), resolve=True),
    pose_noise_scale=[0.001, 0.001, 0.001, 0.001, 0.001, 0.001],  # Per-dim noise
    action_padding=False,
    val_ratio=0.05,
)

# Get a sample
sample = dataset[0]
obs = sample['obs']           # Dictionary with 'camera0_rgb' and 'force'
action = sample['action']     # (T, 13) - action + force delta

print(f"RGB shape: {obs['camera0_rgb'].shape}")    # (T_img, C, H, W)
print(f"Force shape: {obs['force'].shape}")        # (T_force, 6)
print(f"Action shape: {action.shape}")             # (T_action, 13)
```

#### Training Loop Integration

```python
import torch.nn.functional as F

# In your training loop
for batch in dataloader:
    obs = batch['obs']
    action_gt = batch['action']  # (B, T, 13)
    
    # Forward pass
    loss_dict = policy.compute_loss(batch, return_components=True)
    
    # Get loss components
    total_loss = loss_dict['loss']          # Total weighted loss
    action_loss = loss_dict['loss_action']  # Action-only loss
    force_loss = loss_dict['loss_force']    # Force-only loss
    
    # Log to wandb or tensorboard
    logger.log({
        'train/loss': total_loss.item(),
        'train/action_loss': action_loss.item(),
        'train/force_loss': force_loss.item(),
    })
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

#### Customizing the Encoder

```python
from diffusion_policy.model.vision.fadp_encoder import FADPEncoder

# Create custom encoder
encoder = FADPEncoder(
    shape_meta={
        'obs': {
            'camera0_rgb': {'shape': [3, 224, 224], 'type': 'rgb'},
            'force': {'shape': [6], 'type': 'low_dim'}
        }
    },
    rgb_model_name='facebook/dinov3-vit-base-pretrain-lvd1689m',  # HuggingFace model
    use_huggingface=True,      # Use HuggingFace Transformers
    resize_shape=None,         # Processor handles resizing
    force_hidden_dim=128,      # Force MLP hidden dimension
    cross_attn_heads=8,        # Cross-attention heads
    dropout=0.1,
)

# Forward pass
obs_features = encoder(obs)  # (B, feature_dim)
```

### Monitoring Training with WandB

The training workspace automatically logs detailed metrics:

- **Loss Components**: `train_loss_action`, `train_loss_force`, `train_loss_force_weighted`
- **Epoch Averages**: `train_avg_action_loss`, `train_avg_force_loss`
- **Validation Metrics**: `val_loss`, `test_mean_score`
- **Action MSE**: Position, rotation, gripper, and force errors

Example WandB output:
```
train_loss_action: 0.0234
train_loss_force: 0.0156
train_loss_force_weighted: 0.0156  (force_loss_weight * force_loss)
train_loss: 0.0390  (action_loss + force_loss_weighted)
```



