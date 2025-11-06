# Universal Manipulation Interface

A comprehensive framework for robot manipulation learning using diffusion policies. This project provides tools for converting demonstration data, training vision-based manipulation policies, and evaluating model performance.

## Overview

This project implements a complete pipeline for learning manipulation policies from demonstration data:

1. **Data Conversion**: Convert HDF5 demonstration data to zarr format for efficient training
2. **Dataset Support**: Support for FADP (Force-Augmented Diffusion Policy) and UMI (Universal Manipulation Interface) datasets
3. **Training**: Train diffusion-based policies using vision and proprioceptive inputs
4. **Evaluation**: Test trained models and visualize performance

## Features

- **Multi-format Support**: Convert HDF5 session data to zarr format with optimized compression
- **Vision-based Learning**: Support for RGB image inputs with efficient image processing
- **Force/Torque Sensing**: Integration of force/torque data from robot end-effector
- **Flexible Data Augmentation**: Configurable pose noise augmentation for improved robustness
- **Parallel Processing**: Multi-threaded and multi-process support for fast data conversion
- **Comprehensive Inspection Tools**: Utilities for inspecting and validating converted datasets

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd universal_manipulation_interface
```

2. Create and activate conda environment:
```bash
conda env create -f conda_environment.yaml
conda activate umi
```

## Quick Start

### 1. Convert Demonstration Data

Convert your HDF5 session data to zarr format:

```bash
# Basic conversion
python convert_session_to_zarr.py -i data/session_20251025_142256

# With custom options
python convert_session_to_zarr.py \
    -i data/session_20251025_142256 \
    -o output/dataset.zarr.zip \
    -s 224x224 \
    -w 8 \
    --fast-save
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

# Detailed statistics
python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --detailed

# Visualize episode length distribution
python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --visualize
```

### 3. Train a Policy

Train a diffusion policy using the converted data:

```bash
# Train with FADP dataset (vision-only, recommended)
python train.py --config-name=train_diffusion_unet_timm_fadp_workspace \
    task.dataset_path=data/session_20251025_142256/dataset.zarr.zip

# Train with UMI dataset (vision + proprioception)
python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset_path=data/session_20251025_142256/dataset.zarr.zip
```

### 4. Test Trained Model

Evaluate your trained model:

```bash
python test_fadp_model.py --checkpoint <path_to_checkpoint> \
    --dataset-path data/session_20251025_142256/dataset.zarr.zip
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

## Dataset Types

### FADP Dataset (Recommended)

**Force-Augmented Diffusion Policy** - Vision-only learning with force/torque data:

- **Observation**: RGB images only (no proprioception to policy)
- **Action**: 7D relative pose [x, y, z, rx, ry, rz, gripper] in axis-angle representation
- **State Data**: Used internally for computing relative actions, not provided to policy
- **Force Data**: Available for training but not required
- **Data Augmentation**: Gaussian noise on poses during training (configurable)

**Configuration**: `diffusion_policy/config/task/fadp.yaml`

**Training Command**:
```bash
python train.py --config-name=train_diffusion_unet_timm_fadp_workspace
```

### UMI Dataset

**Universal Manipulation Interface** - Full observation learning:

- **Observation**: RGB images + proprioceptive state (position, rotation, gripper)
- **Action**: 7D or 10D depending on rotation representation
- **State Data**: Provided to policy as low-dimensional observations
- **Force Data**: Available optionally

**Configuration**: `diffusion_policy/config/task/umi.yaml`

**Training Command**:
```bash
python train.py --config-name=train_diffusion_unet_timm_umi_workspace
```

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

## Performance Optimization

### Fast Data Conversion

The conversion script supports several optimization options:

1. **Multi-threading**: Parallel image resizing within each episode
   ```bash
   python convert_session_to_zarr.py -i <input> -w 8
   ```

2. **Multi-processing**: Parallel episode processing (faster but uses more memory)
   ```bash
   python convert_session_to_zarr.py -i <input> -w 8 --use-multiprocessing
   ```

3. **Fast Save**: Reduced compression for faster saving (larger files)
   ```bash
   python convert_session_to_zarr.py -i <input> --fast-save
   ```

### Compression Settings

- **Default**: LZ4 compression (level 3) - balanced speed and file size
- **Fast Save**: LZ4 compression (level 1) - fastest, largest files
- **Custom**: Modify compression settings in `convert_session_to_zarr.py`

## Utilities

### Data Inspection

**inspect_zarr.py**: Inspect zarr dataset files
```bash
python inspect_zarr.py <dataset.zarr.zip> [options]
```

**inspect_fadp_dataset.py**: Inspect FADP dataset after loading
```bash
python inspect_fadp_dataset.py --dataset-path <dataset.zarr.zip>
```

### Model Testing

**test_fadp_model.py**: Test trained FADP models
```bash
python test_fadp_model.py \
    --checkpoint <checkpoint_path> \
    --dataset-path <dataset.zarr.zip> \
    --output-dir <output_dir>
```

## Project Structure

```
universal_manipulation_interface/
├── convert_session_to_zarr.py    # Data conversion script
├── train.py                       # Training script
├── test_fadp_model.py            # Model testing script
├── inspect_zarr.py                # Zarr inspection utility
├── inspect_fadp_dataset.py       # FADP dataset inspection
├── diffusion_policy/              # Core diffusion policy framework
│   ├── config/                    # Training configurations
│   │   ├── task/                  # Task-specific configs (fadp.yaml, umi.yaml)
│   │   └── train_*.yaml           # Training workspace configs
│   ├── dataset/                   # Dataset classes
│   │   ├── fadp_dataset.py        # FADP dataset implementation
│   │   └── umi_dataset.py         # UMI dataset implementation
│   ├── policy/                    # Policy implementations
│   ├── model/                     # Model architectures
│   └── workspace/                # Training workspaces
└── data/                          # Data directory
    └── session_*/                 # Session data directories
```

## Troubleshooting

### Common Issues

1. **Memory Error During Conversion**
   - Use `-n` to limit episodes: `python convert_session_to_zarr.py -i <input> -n 10`
   - Reduce `-w` (number of workers)
   - Disable `--use-multiprocessing`

2. **ModuleNotFoundError**
   - Ensure conda environment is activated: `conda activate umi`
   - Check all dependencies are installed

3. **Force Data Missing**
   - The script automatically fills missing force data with zeros
   - Check warnings in conversion output

4. **Slow Conversion**
   - Enable `--fast-save` for faster saving
   - Use `--use-multiprocessing` with multiple workers
   - Increase `-w` for more parallel processing

5. **Dataset Verification**
   - Use `inspect_zarr.py` to verify converted data
   - Check episode counts and data shapes match expectations

## Advanced Usage

### Custom Image Sizes

Resize images for different model architectures:

```bash
# For ViT models (224x224)
python convert_session_to_zarr.py -i <input> -s 224x224

# For larger models (256x256)
python convert_session_to_zarr.py -i <input> -s 256x256
```

### Partial Dataset Conversion

Convert a subset for testing:

```bash
# Convert first 10 episodes
python convert_session_to_zarr.py -i <input> -n 10
```

### Batch Processing

Process multiple sessions:

```bash
for session in data/session_*/; do
    python convert_session_to_zarr.py -i "$session" --fast-save
done
```



