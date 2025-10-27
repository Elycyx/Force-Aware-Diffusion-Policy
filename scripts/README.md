# FADP Data Conversion Scripts

This directory contains utility scripts for converting various data formats to FADP-compatible zarr format.

## 📝 Available Scripts

### `convert_hdf5_to_fadp.py`

Convert HDF5 episode data to FADP zarr format.

**Features:**
- Batch process multiple sessions
- Automatic image resizing
- Data validation
- Progress tracking
- Detailed statistics

## 🚀 Quick Start

### Basic Usage

```bash
python scripts/convert_hdf5_to_fadp.py \
    --input "data/session_*" \
    --output data/fadp_dataset
```

### Advanced Usage

```bash
# Convert with custom image size
python scripts/convert_hdf5_to_fadp.py \
    --input "data/session_20250118_*" \
    --output data/fadp_dataset_96x96 \
    --image-size 96 96

# Process limited episodes for testing
python scripts/convert_hdf5_to_fadp.py \
    --input "data/session_*" \
    --output data/fadp_test \
    --max-episodes 10

# Skip validation for faster processing
python scripts/convert_hdf5_to_fadp.py \
    --input "data/session_*" \
    --output data/fadp_dataset \
    --no-validate
```

## 📋 Input Format Requirements

Your HDF5 episodes must have the following structure:

```
episode_*.hdf5
├── /image              # (N, H, W, 3) uint8
├── /force              # (N, 6) float32
├── /action             # (N, 7) float32
└── /state              # (N, 7) float32 (optional)
```

**Required fields:**
- `image`: RGB images
- `force`: Force/torque measurements (fx, fy, fz, mx, my, mz)
- `action`: Robot actions (dx, dy, dz, drx, dry, drz, gripper)

**Optional fields:**
- `state`: Robot state (not used in FADP)
- `timestamp*`: Various timestamps (not used in FADP)
- `metadata`: Episode metadata

## 📦 Output Format

The script generates a FADP-compatible dataset:

```
output_dir/
├── data.zarr/
│   ├── camera_0/       # Concatenated RGB images
│   ├── force/          # Concatenated force data
│   └── action/         # Concatenated actions
├── meta/
│   └── episode_ends.json    # Episode boundaries
└── conversion_info.json     # Conversion metadata
```

## 🔧 Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input`, `-i` | str | ✅ | Glob pattern for session directories |
| `--output`, `-o` | str | ✅ | Output directory |
| `--image-size` | int int | ❌ | Target image size (H W), default: 240 320 |
| `--max-episodes` | int | ❌ | Max episodes to process, default: all |
| `--no-validate` | flag | ❌ | Skip validation for speed |

## 💡 Tips

### 1. Test First
Always test with a small subset first:
```bash
python scripts/convert_hdf5_to_fadp.py \
    --input "data/session_20250118_143000" \
    --output data/test_output \
    --max-episodes 5
```

### 2. Image Size Selection
Choose image size based on your needs:
- **240×320**: Good balance (default)
- **96×96**: Faster training, lower memory
- **480×640**: Higher quality, slower training

### 3. Memory Considerations
For large datasets (1000+ episodes):
- Process in batches by session
- Use smaller image sizes
- Monitor disk space (zarr is compressed but still large)

### 4. Validation
The script validates:
- ✅ Array shapes and dimensions
- ✅ Data types
- ✅ Value ranges
- ✅ Episode consistency
- ⚠️ Warns about extreme values

## 🐛 Troubleshooting

### Issue: Out of Memory

**Symptom:** Script crashes during concatenation

**Solution:**
```bash
# Process fewer episodes at once
python scripts/convert_hdf5_to_fadp.py \
    --input "data/session_20250118_*" \
    --output data/fadp_part1 \
    --max-episodes 50
```

### Issue: Image Size Mismatch

**Symptom:** Error about image dimensions

**Solution:** Images are automatically resized. Check if source images are valid:
```python
import h5py
with h5py.File('data/session_*/episode0.hdf5', 'r') as f:
    print(f['image'].shape)  # Should be (N, H, W, 3)
```

### Issue: Missing Force Data

**Symptom:** KeyError: 'force'

**Solution:** Ensure your HDF5 files have the `/force` dataset:
```python
import h5py
with h5py.File('episode.hdf5', 'r') as f:
    print(list(f.keys()))  # Should include 'force'
```

### Issue: Wrong Action Dimensions

**Symptom:** AssertionError about action shape

**Solution:** FADP requires 7-DOF actions. If your data has different dimensions:

```python
# Modify the conversion script to handle your action format
# For example, if you have 6-DOF actions, add a dummy gripper:
action_7d = np.zeros((len(action_6d), 7))
action_7d[:, :6] = action_6d
action_7d[:, 6] = 0.5  # Neutral gripper position
```

## 📊 Verifying Output

After conversion, verify the dataset:

```python
import zarr
import json

# Load dataset
root = zarr.open('data/fadp_dataset/data.zarr', mode='r')

print("Camera shape:", root['camera_0'].shape)
print("Force shape:", root['force'].shape)
print("Action shape:", root['action'].shape)

# Load episode boundaries
with open('data/fadp_dataset/meta/episode_ends.json', 'r') as f:
    episode_ends = json.load(f)
    print(f"Number of episodes: {len(episode_ends)}")
    print(f"Total frames: {episode_ends[-1]}")
```

Or use the validation script:

```bash
python -c "
from DATASET_FORMAT import validate_fadp_dataset
validate_fadp_dataset('data/fadp_dataset')
"
```

## 📚 Next Steps

After conversion:

1. **Verify the dataset** (see above)
2. **Update your task config**:
   ```yaml
   # fadp/config/task/my_task.yaml
   dataset_path: data/fadp_dataset
   image_shape: [3, 240, 320]  # Match your conversion
   ```
3. **Train the model**:
   ```bash
   python train.py --config-name=train_force_aware_diffusion_policy_real
   ```

## 🤝 Contributing

To add support for other data formats:

1. Create a new conversion script (e.g., `convert_rosbag_to_fadp.py`)
2. Follow the same output format
3. Add documentation here
4. Submit a pull request

## 📖 See Also

- [DATASET_FORMAT.md](../DATASET_FORMAT.md) - Detailed dataset format specification
- [README.md](../README.md) - Main project documentation
- [FORCE_DATA_FLOW.md](../FORCE_DATA_FLOW.md) - How force data flows through the system

