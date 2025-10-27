# FADP Dataset Format Specification

This document describes the required data format for training Force-Aware Diffusion Policy (FADP).

## 📁 Directory Structure

```
your_dataset/
├── data.zarr/              # Main data store (zarr format)
│   ├── camera_0/           # RGB images from camera 0
│   ├── camera_1/           # RGB images from camera 1 (optional)
│   ├── force/              # Force/torque sensor readings
│   └── action/             # Robot actions
└── meta/
    └── episode_ends.json   # Episode boundary information
```

## 📊 Data Components

### 1. RGB Images (`camera_0`, `camera_1`, ...)

**Format:** Zarr array  
**Shape:** `(total_timesteps, height, width, 3)`  
**Dtype:** `uint8`  
**Value Range:** `[0, 255]`

**Description:**
- RGB images from robot-mounted cameras
- Images are stored in HWC format (Height × Width × Channels)
- Channel order: RGB (not BGR)
- Multiple cameras supported (camera_0, camera_1, etc.)

**Example:**
```python
# Shape: (1000, 240, 320, 3) for 1000 frames at 240×320 resolution
camera_data = zarr_root['camera_0']
print(camera_data.shape)  # (1000, 240, 320, 3)
print(camera_data.dtype)  # uint8
```

**Typical Resolutions:**
- `240 × 320` (common for RealSense)
- `480 × 640` (higher quality)
- `96 × 96` (for training efficiency)

### 2. Force/Torque Data (`force`)

**Format:** Zarr array  
**Shape:** `(total_timesteps, 6)`  
**Dtype:** `float32`  
**Units:** Forces in Newtons (N), Torques in Newton-meters (N·m)

**Description:**
- 6-DOF force/torque measurements from sensor (e.g., ATI, Robotiq FT300)
- **Column order is critical:** `[fx, fy, fz, mx, my, mz]`

**Column Definitions:**
```
Index | Name | Description                    | Unit
------|------|--------------------------------|------
  0   | fx   | Force along X-axis             | N
  1   | fy   | Force along Y-axis             | N
  2   | fz   | Force along Z-axis             | N
  3   | mx   | Torque around X-axis (roll)    | N·m
  4   | my   | Torque around Y-axis (pitch)   | N·m
  5   | mz   | Torque around Z-axis (yaw)     | N·m
```

**Coordinate Frame:**
- Forces and torques are measured in the sensor's local frame
- Ensure consistent frame definition across your dataset
- Document your coordinate convention (e.g., right-hand rule)

**Example:**
```python
# Shape: (1000, 6) for 1000 timesteps
force_data = zarr_root['force']
print(force_data.shape)      # (1000, 6)
print(force_data.dtype)      # float32

# Sample reading
sample = force_data[0]
print(f"fx={sample[0]:.2f}N, fy={sample[1]:.2f}N, fz={sample[2]:.2f}N")
print(f"mx={sample[3]:.3f}N·m, my={sample[4]:.3f}N·m, mz={sample[5]:.3f}N·m")
```

**Typical Value Ranges:**
- Forces: `-50N to +50N` (depends on sensor and task)
- Torques: `-5N·m to +5N·m` (depends on sensor and task)

**Important Notes:**
- ⚠️ Remove sensor bias/offset before saving
- ⚠️ Apply gravity compensation if needed
- ⚠️ Filter high-frequency noise (e.g., 50Hz low-pass filter)
- ⚠️ Ensure data is synchronized with images and actions

### 3. Actions (`action`)

**Format:** Zarr array  
**Shape:** `(total_timesteps, 7)`  
**Dtype:** `float32`

**Description:**
- Robot action commands
- 7-DOF: delta end-effector pose + gripper

**Column Definitions:**
```
Index | Name    | Description                      | Unit
------|---------|----------------------------------|--------
  0   | dx      | Delta position along X-axis      | meters
  1   | dy      | Delta position along Y-axis      | meters
  2   | dz      | Delta position along Z-axis      | meters
  3   | drx     | Delta rotation around X-axis     | radians
  4   | dry     | Delta rotation around Y-axis     | radians
  5   | drz     | Delta rotation around Z-axis     | radians
  6   | gripper | Gripper command (0=open, 1=close)| [0, 1]
```

**Example:**
```python
# Shape: (1000, 7) for 1000 timesteps
action_data = zarr_root['action']
print(action_data.shape)     # (1000, 7)
print(action_data.dtype)     # float32

# Sample action
sample = action_data[0]
print(f"Position delta: ({sample[0]:.3f}, {sample[1]:.3f}, {sample[2]:.3f})m")
print(f"Rotation delta: ({sample[3]:.3f}, {sample[4]:.3f}, {sample[5]:.3f})rad")
print(f"Gripper: {sample[6]:.2f}")
```

**Typical Value Ranges:**
- Position deltas: `-0.05m to +0.05m` per step
- Rotation deltas: `-0.2rad to +0.2rad` per step
- Gripper: `0.0` (fully open) to `1.0` (fully closed)

### 4. Episode Boundaries (`meta/episode_ends.json`)

**Format:** JSON file  
**Content:** List of integers representing the ending index of each episode

**Description:**
- Defines where each demonstration episode ends
- Used to prevent cross-episode sampling during training

**Example:**
```json
[150, 320, 495, 680, 850, 1000]
```

This means:
- Episode 0: frames 0-149 (150 frames)
- Episode 1: frames 150-319 (170 frames)
- Episode 2: frames 320-494 (175 frames)
- Episode 3: frames 495-679 (185 frames)
- Episode 4: frames 680-849 (170 frames)
- Episode 5: frames 850-999 (150 frames)

## 🔧 Creating a Dataset

### Method 1: From Raw Data

```python
import zarr
import numpy as np
import json
from pathlib import Path

def create_fadp_dataset(
    output_dir: str,
    rgb_images: np.ndarray,      # (T, H, W, 3)
    force_data: np.ndarray,       # (T, 6)
    actions: np.ndarray,          # (T, 7)
    episode_ends: list,           # [150, 300, ...]
):
    """
    Create a FADP-compatible dataset.
    
    Args:
        output_dir: Path to output directory
        rgb_images: RGB images (uint8, 0-255)
        force_data: Force/torque readings (float32)
        actions: Action commands (float32)
        episode_ends: List of episode ending indices
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create zarr store
    store_path = output_path / 'data.zarr'
    store = zarr.DirectoryStore(store_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Save data
    print(f"Saving camera data: {rgb_images.shape}")
    root.create_dataset('camera_0', 
                       data=rgb_images, 
                       dtype='uint8',
                       chunks=(1, rgb_images.shape[1], rgb_images.shape[2], 3))
    
    print(f"Saving force data: {force_data.shape}")
    root.create_dataset('force', 
                       data=force_data, 
                       dtype='float32',
                       chunks=(100, 6))
    
    print(f"Saving action data: {actions.shape}")
    root.create_dataset('action', 
                       data=actions, 
                       dtype='float32',
                       chunks=(100, 7))
    
    # Save episode metadata
    meta_dir = output_path / 'meta'
    meta_dir.mkdir(exist_ok=True)
    
    with open(meta_dir / 'episode_ends.json', 'w') as f:
        json.dump(episode_ends, f)
    
    print(f"Dataset created at: {output_path}")
    print(f"Total timesteps: {len(rgb_images)}")
    print(f"Number of episodes: {len(episode_ends)}")

# Example usage
if __name__ == "__main__":
    # Load your data
    images = np.random.randint(0, 255, (1000, 240, 320, 3), dtype=np.uint8)
    forces = np.random.randn(1000, 6).astype(np.float32) * 5.0  # ±5N/N·m
    actions = np.random.randn(1000, 7).astype(np.float32) * 0.01  # Small deltas
    actions[:, 6] = np.random.rand(1000)  # Gripper [0, 1]
    
    episode_ends = [200, 400, 600, 800, 1000]
    
    create_fadp_dataset(
        output_dir='data/my_robot_dataset',
        rgb_images=images,
        force_data=forces,
        actions=actions,
        episode_ends=episode_ends
    )
```

### Method 2: From ROS Bag (Example)

```python
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np

def rosbag_to_fadp_dataset(
    bag_path: str,
    output_dir: str,
    camera_topic: str = '/camera/color/image_raw',
    force_topic: str = '/force_torque_sensor',
    action_topic: str = '/robot/action',
):
    """Convert ROS bag to FADP dataset format"""
    
    bridge = CvBridge()
    bag = rosbag.Bag(bag_path)
    
    images = []
    forces = []
    actions = []
    
    # Read messages
    for topic, msg, t in bag.read_messages(topics=[camera_topic, force_topic, action_topic]):
        if topic == camera_topic:
            cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
            images.append(cv_image)
        
        elif topic == force_topic:
            # Assuming geometry_msgs/WrenchStamped
            force = [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ]
            forces.append(force)
        
        elif topic == action_topic:
            # Parse your action message format
            action = parse_action_msg(msg)
            actions.append(action)
    
    bag.close()
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    forces = np.array(forces, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    
    # Create dataset
    create_fadp_dataset(
        output_dir=output_dir,
        rgb_images=images,
        force_data=forces,
        actions=actions,
        episode_ends=[len(images)]  # Single episode
    )
```

## ✅ Dataset Validation

After creating your dataset, validate it:

```python
import zarr
import json
import matplotlib.pyplot as plt

def validate_fadp_dataset(dataset_path: str):
    """Validate dataset format and show statistics"""
    
    # Load data
    store = zarr.DirectoryStore(f'{dataset_path}/data.zarr')
    root = zarr.group(store=store)
    
    # Load episode ends
    with open(f'{dataset_path}/meta/episode_ends.json', 'r') as f:
        episode_ends = json.load(f)
    
    print("="*60)
    print("FADP Dataset Validation")
    print("="*60)
    
    # Check camera data
    print("\n📷 Camera Data:")
    if 'camera_0' in root:
        cam_shape = root['camera_0'].shape
        print(f"  Shape: {cam_shape}")
        print(f"  Dtype: {root['camera_0'].dtype}")
        print(f"  Value range: [{root['camera_0'][:].min()}, {root['camera_0'][:].max()}]")
        assert cam_shape[-1] == 3, "Camera should have 3 channels (RGB)"
        assert root['camera_0'].dtype == np.uint8, "Camera should be uint8"
    else:
        print("  ⚠️  No camera_0 found!")
    
    # Check force data
    print("\n💪 Force Data:")
    if 'force' in root:
        force_shape = root['force'].shape
        print(f"  Shape: {force_shape}")
        print(f"  Dtype: {root['force'].dtype}")
        assert force_shape[-1] == 6, "Force should have 6 dimensions"
        assert root['force'].dtype == np.float32, "Force should be float32"
        
        force_data = root['force'][:]
        print(f"\n  Statistics per dimension:")
        labels = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
        for i, label in enumerate(labels):
            mean = force_data[:, i].mean()
            std = force_data[:, i].std()
            min_val = force_data[:, i].min()
            max_val = force_data[:, i].max()
            print(f"    {label}: mean={mean:+.3f}, std={std:.3f}, range=[{min_val:+.3f}, {max_val:+.3f}]")
    else:
        print("  ⚠️  No force data found!")
    
    # Check action data
    print("\n🎮 Action Data:")
    if 'action' in root:
        action_shape = root['action'].shape
        print(f"  Shape: {action_shape}")
        print(f"  Dtype: {root['action'].dtype}")
        assert action_shape[-1] == 7, "Action should have 7 dimensions"
        assert root['action'].dtype == np.float32, "Action should be float32"
        
        action_data = root['action'][:]
        print(f"\n  Statistics per dimension:")
        labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
        for i, label in enumerate(labels):
            mean = action_data[:, i].mean()
            std = action_data[:, i].std()
            min_val = action_data[:, i].min()
            max_val = action_data[:, i].max()
            print(f"    {label}: mean={mean:+.4f}, std={std:.4f}, range=[{min_val:+.4f}, {max_val:+.4f}]")
    else:
        print("  ⚠️  No action data found!")
    
    # Check episodes
    print("\n📊 Episode Information:")
    print(f"  Number of episodes: {len(episode_ends)}")
    print(f"  Total timesteps: {episode_ends[-1]}")
    
    episode_lengths = []
    for i, end in enumerate(episode_ends):
        start = 0 if i == 0 else episode_ends[i-1]
        length = end - start
        episode_lengths.append(length)
    
    print(f"  Episode lengths: min={min(episode_lengths)}, max={max(episode_lengths)}, mean={np.mean(episode_lengths):.1f}")
    
    # Visualization
    print("\n📈 Generating visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot force data
    force_data = root['force'][:]
    for i in range(6):
        ax = axes[i//2, i%2]
        ax.plot(force_data[:, i])
        ax.set_title(labels[i])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_path}_validation.png')
    print(f"  Saved visualization to: {dataset_path}_validation.png")
    
    print("\n✅ Validation complete!")

# Usage
validate_fadp_dataset('data/my_robot_dataset')
```

## 📝 Configuration File

Update your task configuration to match your dataset:

```yaml
# fadp/config/task/my_robot_task.yaml

name: my_robot_task
image_shape: [3, 240, 320]  # Match your camera resolution
dataset_path: data/my_robot_dataset

shape_meta: &shape_meta
  obs:
    camera_0:
      shape: ${task.image_shape}
      type: rgb
    force:
      shape: [6]
      type: low_dim
  action:
    shape: [7]

dataset:
  _target_: fadp.dataset.real_pusht_image_dataset.RealPushTImageDataset
  shape_meta: *shape_meta
  dataset_path: ${task.dataset_path}
  horizon: 16
  pad_before: 1
  pad_after: 7
  n_obs_steps: 2
  use_cache: True
  seed: 42
  val_ratio: 0.05
```

## ⚠️ Common Issues

### Issue 1: Mismatched Dimensions
```
Error: Expected force shape (T, 6), got (T, 3)
```
**Solution:** Ensure your force data includes all 6 components (fx, fy, fz, mx, my, mz)

### Issue 2: Wrong Data Type
```
Error: Expected uint8 for images, got float64
```
**Solution:** Convert images to uint8 with values in [0, 255]

### Issue 3: Unsynchronized Data
```
Error: camera_0 has 1000 frames but force has 998
```
**Solution:** Ensure all modalities have the same number of timesteps

### Issue 4: Missing Episode Ends
```
Error: Episode ends [1000] doesn't match data length 995
```
**Solution:** Last episode end must equal total number of timesteps

## 📚 Additional Resources

- **Example Dataset:** See `data/example_dataset/` (if provided)
- **Data Collection Script:** `scripts/collect_data.py` (if provided)
- **Data Visualization:** `scripts/visualize_dataset.py` (if provided)

## 💡 Tips

1. **Synchronization:** Ensure camera, force sensor, and robot commands are time-synchronized
2. **Frequency:** Collect data at consistent frequency (e.g., 10Hz, 30Hz)
3. **Calibration:** Calibrate force sensor and remove bias before collection
4. **Variety:** Collect diverse demonstrations for better generalization
5. **Quality:** Remove failed demonstrations with errors or collisions
6. **Compression:** Use zarr chunking for efficient storage and loading

---

For questions about dataset format, please open an issue on GitHub.

