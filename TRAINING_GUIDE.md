# FADP Training Guide

This guide explains how to train Force-Aware Diffusion Policy with your converted dataset.

## 📋 Prerequisites

Before training, ensure you have:

1. ✅ Converted your HDF5 data to FADP format (see `scripts/README.md`)
2. ✅ Validated the dataset format (see `DATASET_FORMAT.md`)
3. ✅ Installed all dependencies (`conda env create -f conda_environment.yaml`)
4. ✅ Understand the difference between datasets (see [Dataset Choice](#-dataset-choice) below)

## 📦 Dataset Choice

FADP provides two dataset classes:

### FADPDataset (Recommended) ⭐

- **Modern design**: Built specifically for FADP
- **Relative actions**: Learns delta from first state in action chunk
- **Efficient**: Direct zarr loading without intermediate buffers
- **Simple configuration**: Direct parameter specification
- **Use case**: New projects, better generalization needed

See [FADP_DATASET.md](FADP_DATASET.md) for detailed documentation.

### RealPushTImageDataset (Legacy)

- **Compatibility**: Works with existing robomimic-style data
- **Absolute actions**: Learns absolute target positions
- **ReplayBuffer**: Uses intermediate buffer for flexibility
- **Complex configuration**: Requires shape_meta parsing
- **Use case**: Existing projects, absolute positioning needed

## 🚀 Quick Start

### Option 1: Using FADPDataset (Recommended)

#### Step 1: Prepare Your Dataset

After converting with `convert_hdf5_to_fadp.py`, you should have:

```
data/fadp_dataset/
├── data.zarr/
│   ├── camera_0/
│   ├── force/
│   └── action/
└── meta/
    └── episode_ends.json
```

#### Step 2: Test the Dataset

```bash
python test_fadp_dataset.py --dataset data/fadp_dataset --compare
```

This will verify:
- Dataset loads correctly
- Relative actions are computed properly
- Normalization works
- Compare relative vs absolute action representations

#### Step 3: Start Training

```bash
python train.py --config-name=train_fadp task.dataset.dataset_path=data/fadp_dataset
```

That's it! The `train_fadp.yaml` configuration is already set up for FADPDataset.

### Option 2: Using RealPushTImageDataset (Legacy)

#### Step 1: Prepare Your Dataset

Same as Option 1.

#### Step 2: Create Task Configuration

Create a task config file at `fadp/config/task/my_robot_task.yaml`:

```yaml
name: my_robot_task
image_shape: [3, 240, 320]  # Match your converted image size
dataset_path: data/fadp_dataset  # Path to your dataset

shape_meta: &shape_meta
  # Observation definition
  obs:
    camera_0:
      shape: ${task.image_shape}
      type: rgb
    force:
      shape: [6]  # fx, fy, fz, mx, my, mz
      type: low_dim
  
  # Action definition
  action:
    shape: [7]  # dx, dy, dz, drx, dry, drz, gripper

# Environment runner (optional, for evaluation during training)
env_runner:
  _target_: fadp.env_runner.real_pusht_image_runner.RealPushTImageRunner

# Dataset configuration
dataset:
  _target_: fadp.dataset.real_pusht_image_dataset.RealPushTImageDataset
  shape_meta: *shape_meta
  dataset_path: ${task.dataset_path}
  horizon: 16              # Prediction horizon
  pad_before: 1            # Context padding
  pad_after: 7             # Future padding
  n_obs_steps: 2           # Number of observation frames
  n_latency_steps: 0       # Action latency compensation
  use_cache: True          # Cache dataset for faster loading
  seed: 42
  val_ratio: 0.05          # 5% validation split
  max_train_episodes: null # Use all episodes
  delta_action: False      # Use absolute actions
```

### Step 3: Create Training Configuration

Create `fadp/config/train_my_robot.yaml`:

```yaml
defaults:
  - _self_
  - task: my_robot_task  # Reference your task config

name: train_fadp_my_robot
_target_: fadp.workspace.train_force_aware_diffusion_policy_workspace.TrainForceAwareDiffusionPolicyWorkspace

task_name: ${task.name}
shape_meta: ${task.shape_meta}
exp_name: "my_first_experiment"

# Training parameters
horizon: 16           # How many steps to predict
n_obs_steps: 2        # How many observation frames to use
n_action_steps: 8     # How many actions to execute
n_latency_steps: 0    # Latency compensation
dataset_obs_steps: ${n_obs_steps}
past_action_visible: False
obs_as_global_cond: True  # Use observations as global conditioning

# Policy architecture
policy:
  _target_: fadp.policy.force_aware_diffusion_policy.ForceAwareDiffusionPolicy
  
  shape_meta: ${shape_meta}
  
  # Diffusion scheduler
  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_ddim.DDIMScheduler
    num_train_timesteps: 100
    beta_start: 0.0001
    beta_end: 0.02
    beta_schedule: squaredcos_cap_v2
    clip_sample: True
    set_alpha_to_one: True
    steps_offset: 0
    prediction_type: epsilon
  
  # Model parameters
  horizon: ${horizon}
  n_action_steps: ${n_action_steps}
  n_obs_steps: ${n_obs_steps}
  num_inference_steps: 16  # DDIM steps during inference
  obs_as_global_cond: ${obs_as_global_cond}
  
  # Vision encoder settings
  crop_shape: [216, 288]  # Crop size for images [H, W]
  obs_encoder_group_norm: True
  eval_fixed_crop: True
  
  # Diffusion model architecture
  diffusion_step_embed_dim: 128
  down_dims: [256, 512, 1024]
  kernel_size: 5
  n_groups: 8
  cond_predict_scale: True
  
  # Force encoder settings
  force_encoder_hidden_dim: 512  # Hidden layer size for force MLP

# EMA (Exponential Moving Average) for stable training
ema:
  _target_: fadp.model.diffusion.ema_model.EMAModel
  update_after_step: 0
  inv_gamma: 1.0
  power: 0.75
  min_value: 0.0
  max_value: 0.9999

# Data loader settings
dataloader:
  batch_size: 64      # Adjust based on GPU memory
  num_workers: 8      # Parallel data loading
  shuffle: True
  pin_memory: True
  persistent_workers: True

val_dataloader:
  batch_size: 64
  num_workers: 8
  shuffle: False
  pin_memory: True
  persistent_workers: True

# Optimizer
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4          # Learning rate
  betas: [0.95, 0.999]
  eps: 1.0e-8
  weight_decay: 1.0e-6

# Training schedule
training:
  device: "cuda:0"    # GPU device
  seed: 42
  debug: False
  resume: True        # Resume from checkpoint if exists
  
  # Learning rate schedule
  lr_scheduler: cosine
  lr_warmup_steps: 500
  num_epochs: 600     # Total training epochs
  gradient_accumulate_every: 1
  use_ema: True       # Use EMA model
  
  # Evaluation and checkpointing
  rollout_every: 50   # Evaluate every N epochs (if env_runner available)
  checkpoint_every: 50
  val_every: 1        # Validate every epoch
  sample_every: 5     # Sample predictions for visualization
  
  # Training steps
  max_train_steps: null
  max_val_steps: null
  
  tqdm_interval_sec: 1.0

# Weights & Biases logging
logging:
  project: fadp_experiments
  resume: True
  mode: online  # "online", "offline", or "disabled"
  name: ${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}
  tags: ["${name}", "${task_name}", "${exp_name}"]
  id: null
  group: null

# Checkpoint management
checkpoint:
  topk:
    monitor_key: train_loss
    mode: min
    k: 5  # Keep best 5 checkpoints
    format_str: 'epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt'
  save_last_ckpt: True
  save_last_snapshot: False

# Multi-run settings
multi_run:
  run_dir: data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
  wandb_name_base: ${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}

# Hydra configuration
hydra:
  job:
    override_dirname: ${name}
  run:
    dir: data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
  sweep:
    dir: data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
    subdir: ${hydra.job.num}
```

### Step 4: Start Training

```bash
# Basic training
python train.py --config-name=train_my_robot

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py --config-name=train_my_robot

# Override parameters
python train.py --config-name=train_my_robot \
    training.batch_size=32 \
    training.num_epochs=1000 \
    optimizer.lr=5e-5
```

## 📊 Monitor Training

### Weights & Biases (Recommended)

Training metrics are automatically logged to W&B:

1. First time: `wandb login`
2. View at: `https://wandb.ai/your-username/fadp_experiments`

Logged metrics:
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `train_action_mse_error`: Action prediction error
- `lr`: Current learning rate

### Local Logs

Logs are also saved locally:

```
data/outputs/YYYY.MM.DD/HH.MM.SS_train_fadp_my_robot_my_robot_task/
├── checkpoints/
│   ├── latest.ckpt
│   └── epoch=0100-train_loss=0.123.ckpt
├── logs.json.txt  # Training logs in JSON format
└── .hydra/
    └── config.yaml  # Full configuration used
```

View logs:
```bash
# Real-time monitoring
tail -f data/outputs/*/logs.json.txt

# Parse with jq
cat data/outputs/*/logs.json.txt | jq -r '[.epoch, .train_loss, .val_loss] | @tsv'
```

## 🔧 Common Training Issues

### Issue 1: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```yaml
dataloader:
  batch_size: 32  # Try 32, 16, or 8
```

2. **Reduce image size:**
```yaml
task:
  image_shape: [3, 96, 96]  # Smaller images
policy:
  crop_shape: [84, 84]
```

3. **Use gradient accumulation:**
```yaml
training:
  gradient_accumulate_every: 2  # Effective batch_size = batch_size * 2
dataloader:
  batch_size: 32
```

4. **Disable EMA temporarily:**
```yaml
training:
  use_ema: False
```

### Issue 2: Training Loss Not Decreasing

**Possible causes:**

1. **Learning rate too high/low:**
```yaml
optimizer:
  lr: 5.0e-5  # Try different values: 1e-4, 5e-5, 1e-5
```

2. **Not enough data:**
- Collect more demonstrations (aim for 100+ episodes)
- Increase data augmentation (implicit in crop_shape)

3. **Check data normalization:**
```python
# Verify your data ranges
import zarr
root = zarr.open('data/fadp_dataset/data.zarr', 'r')
print("Force range:", root['force'][:].min(), root['force'][:].max())
print("Action range:", root['action'][:].min(), root['action'][:].max())
```

### Issue 3: Validation Loss Increasing (Overfitting)

**Solutions:**

1. **Increase validation split:**
```yaml
dataset:
  val_ratio: 0.1  # Use 10% for validation
```

2. **Reduce model capacity:**
```yaml
policy:
  down_dims: [128, 256, 512]  # Smaller network
  force_encoder_hidden_dim: 256
```

3. **Add more data or stop training earlier:**
```yaml
training:
  num_epochs: 300  # Stop earlier
```

### Issue 4: Data Loading Slow

**Solutions:**

1. **Enable caching:**
```yaml
dataset:
  use_cache: True  # Cache processed data
```

2. **Increase workers:**
```yaml
dataloader:
  num_workers: 16  # More parallel loading
```

3. **Use faster storage:**
- Move dataset to SSD
- Use RAM disk for very small datasets

## 📈 Training Tips

### 1. Start with Small Experiments

Test with limited data first:

```yaml
dataset:
  max_train_episodes: 10  # Use only 10 episodes
training:
  num_epochs: 10
  debug: True  # Faster iterations
```

### 2. Monitor Key Metrics

Good training should show:
- ✅ `train_loss` steadily decreasing
- ✅ `val_loss` close to `train_loss`
- ✅ `train_action_mse_error` decreasing

Warning signs:
- ⚠️ `val_loss >> train_loss`: Overfitting
- ⚠️ Loss plateaus early: Learning rate too low
- ⚠️ Loss explodes: Learning rate too high

### 3. Hyperparameter Tuning Priority

Tune in this order:

1. **Batch size**: As large as GPU allows (32-64)
2. **Learning rate**: 1e-4 is good default, try 5e-5 to 2e-4
3. **Horizon**: Match your task requirements (16 is good default)
4. **n_action_steps**: 8-16 typically works well
5. **Force encoder**: 256-512 hidden dim

### 4. Data Quality Matters Most

Before extensive training:
- ✅ Verify force sensor calibration
- ✅ Check for outliers in force data
- ✅ Ensure consistent robot behavior
- ✅ Remove failed demonstrations

### 5. Checkpoint Best Models

```yaml
checkpoint:
  topk:
    monitor_key: val_loss  # Save based on validation loss
    mode: min
    k: 10  # Keep top 10 models
```

## 🎯 After Training

### Step 1: Find Best Checkpoint

```bash
# List checkpoints sorted by loss
ls -lh data/outputs/*/checkpoints/*.ckpt | sort
```

### Step 2: Test on Robot

See evaluation guide:

```bash
python eval_real_robot.py \
    -i data/outputs/.../checkpoints/best_model.ckpt \
    -o eval_results \
    --robot_ip 192.168.1.100
```

### Step 3: Iterate

If performance is poor:
1. Collect more diverse data
2. Check force sensor calibration
3. Tune hyperparameters
4. Increase training time

## 📚 Advanced Topics

### Multi-GPU Training

```bash
# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config-name=train_my_robot \
    training.device="cuda"
```

### Resume Training

Training automatically resumes if checkpoint exists:

```yaml
training:
  resume: True  # Default
```

Manually specify checkpoint:
```bash
python train.py --config-name=train_my_robot \
    resume_checkpoint=path/to/checkpoint.ckpt
```

### Debug Mode

Fast iterations for debugging:

```yaml
training:
  debug: True
  max_train_steps: 3
  max_val_steps: 3
  num_epochs: 2
```

### Sweep Hyperparameters

See `ray_train_multirun.py` for hyperparameter sweeps.

## ✅ Checklist Before Training

- [ ] Dataset converted and validated
- [ ] Task config created with correct shapes
- [ ] Training config created
- [ ] GPU available and CUDA working
- [ ] W&B account setup (optional but recommended)
- [ ] Enough disk space for checkpoints (few GB)
- [ ] Small test run successful

## 🆘 Getting Help

If training fails:

1. Check logs: `data/outputs/*/logs.json.txt`
2. Verify data: See `DATASET_FORMAT.md`
3. Test model: Run `test_fadp_model.py`
4. Open GitHub issue with:
   - Error message
   - Config file
   - System info (`nvidia-smi`, `python --version`)

## 📖 Related Documentation

- [README.md](README.md) - Main documentation
- [DATASET_FORMAT.md](DATASET_FORMAT.md) - Dataset requirements
- [FORCE_DATA_FLOW.md](FORCE_DATA_FLOW.md) - How force data is processed
- [scripts/README.md](scripts/README.md) - Data conversion

---

Happy training! 🚀

