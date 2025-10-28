"""
FADP Dataset - Force-Aware Diffusion Policy Dataset

This dataset is specifically designed for FADP with the following features:
- Supports RGB images + force/torque sensor data
- Uses RELATIVE actions (delta from first state in action chunk)
- Loads data from zarr format
- Efficient caching and normalization
"""

from typing import Dict, Optional
import torch
import numpy as np
import zarr
import os
from pathlib import Path
import json
import copy
from threadpoolctl import threadpool_limits
from scipy.spatial.transform import Rotation as R

from fadp.common.pytorch_util import dict_apply
from fadp.dataset.base_dataset import BaseImageDataset
from fadp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from fadp.common.normalize_util import get_image_range_normalizer


class FADPDataset(BaseImageDataset):
    """
    Force-Aware Diffusion Policy Dataset
    
    Key features:
    - Loads RGB images, force data, and actions from zarr
    - Converts actions to RELATIVE representation (delta from first state)
    - Efficient sequence sampling with proper padding
    - Automatic normalization
    """
    
    def __init__(
        self,
        dataset_path: str,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        image_keys: list = None,
        force_key: str = 'force',
        action_key: str = 'action',
        use_relative_action: bool = True,
        use_reference_noise: bool = True,
        reference_noise_scale: list = None,
        val_ratio: float = 0.0,
        seed: int = 42,
    ):
        """
        Args:
            dataset_path: Path to zarr dataset directory
            horizon: Total prediction horizon
            n_obs_steps: Number of observation frames to use as context
            n_action_steps: Number of action steps to execute
            image_keys: List of image observation keys (e.g., ['camera_0'])
            force_key: Key for force/torque data
            action_key: Key for action data
            use_relative_action: If True, convert actions to relative (delta from first)
            use_reference_noise: If True, add noise to reference pose for data augmentation
            reference_noise_scale: Noise scale [x, y, z, rx, ry, rz] (default: [0.01, 0.01, 0.01, 0.02, 0.02, 0.02])
            val_ratio: Validation split ratio
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.dataset_path = Path(dataset_path)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.use_relative_action = use_relative_action
        self.use_reference_noise = use_reference_noise
        self.val_ratio = val_ratio
        self.seed = seed
        
        # Set reference noise scale
        if reference_noise_scale is None:
            self.reference_noise_scale = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
        else:
            self.reference_noise_scale = np.array(reference_noise_scale)
        
        # Default image keys
        if image_keys is None:
            image_keys = ['camera_0']
        self.image_keys = image_keys
        self.force_key = force_key
        self.action_key = action_key
        
        # Load data
        print(f"Loading FADP dataset from: {self.dataset_path}")
        self._load_data()
        
        # Split train/val
        self._create_split()
        
        print(f"Dataset loaded:")
        print(f"  Total episodes: {self.n_episodes}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Train episodes: {self.train_mask.sum()}")
        print(f"  Val episodes: {self.val_mask.sum()}")
        print(f"  Relative actions: {self.use_relative_action}")
        if self.use_relative_action:
            print(f"  Reference noise: {self.use_reference_noise}")
            if self.use_reference_noise:
                print(f"  Noise scale: {self.reference_noise_scale}")
    
    def _load_data(self):
        """Load data from zarr store"""
        zarr_path = self.dataset_path / 'data.zarr'
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
        
        # Open zarr store
        self.zarr_root = zarr.open(str(zarr_path), mode='r')
        
        # Load episode boundaries
        meta_path = self.dataset_path / 'meta' / 'episode_ends.json'
        with open(meta_path, 'r') as f:
            self.episode_ends = np.array(json.load(f))
        
        self.n_episodes = len(self.episode_ends)
        self.total_frames = self.episode_ends[-1]
        
        # Validate data
        self._validate_data()
        
        # Create episode start indices
        self.episode_starts = np.zeros(self.n_episodes, dtype=np.int64)
        self.episode_starts[1:] = self.episode_ends[:-1]
    
    def _validate_data(self):
        """Validate that all required data exists and has correct shape"""
        # Check images
        for key in self.image_keys:
            if key not in self.zarr_root:
                raise KeyError(f"Image key '{key}' not found in dataset")
            img_shape = self.zarr_root[key].shape
            if img_shape[0] != self.total_frames:
                raise ValueError(f"Image '{key}' has {img_shape[0]} frames, expected {self.total_frames}")
        
        # Check force
        if self.force_key not in self.zarr_root:
            raise KeyError(f"Force key '{self.force_key}' not found in dataset")
        force_shape = self.zarr_root[self.force_key].shape
        if force_shape != (self.total_frames, 6):
            raise ValueError(f"Force data has shape {force_shape}, expected ({self.total_frames}, 6)")
        
        # Check action
        if self.action_key not in self.zarr_root:
            raise KeyError(f"Action key '{self.action_key}' not found in dataset")
        action_shape = self.zarr_root[self.action_key].shape
        if action_shape[0] != self.total_frames:
            raise ValueError(f"Action has {action_shape[0]} frames, expected {self.total_frames}")
        
        self.action_dim = action_shape[1]
        print(f"  Action dimension: {self.action_dim}")
    
    def _create_split(self):
        """Create train/validation split"""
        np.random.seed(self.seed)
        
        # Random split by episode
        n_val = int(self.n_episodes * self.val_ratio)
        val_indices = np.random.choice(
            self.n_episodes,
            size=n_val,
            replace=False
        )
        
        self.val_mask = np.zeros(self.n_episodes, dtype=bool)
        self.val_mask[val_indices] = True
        self.train_mask = ~self.val_mask
    
    def _get_valid_indices(self, episode_mask: np.ndarray) -> np.ndarray:
        """
        Get valid sample start indices for given episode mask.
        An index is valid if we can sample a full sequence from it.
        """
        valid_indices = []
        
        for ep_idx in np.where(episode_mask)[0]:
            start = self.episode_starts[ep_idx]
            end = self.episode_ends[ep_idx]
            
            # We need at least horizon frames to sample from this episode
            if end - start >= self.horizon:
                # Valid start indices: from start to (end - horizon)
                episode_valid = np.arange(start, end - self.horizon + 1)
                valid_indices.append(episode_valid)
        
        if len(valid_indices) == 0:
            raise ValueError("No valid indices found! Check episode lengths and horizon.")
        
        valid_indices = np.concatenate(valid_indices)
        return valid_indices
    
    def get_validation_dataset(self):
        """Get validation split"""
        val_dataset = copy.copy(self)
        val_dataset.train_mask = self.val_mask
        val_dataset.val_mask = self.train_mask
        return val_dataset
    
    def __len__(self):
        """Return number of valid samples"""
        valid_indices = self._get_valid_indices(self.train_mask)
        return len(valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample a sequence from the dataset.
        
        Returns:
            dict with keys:
                - 'obs': dict containing RGB images and force data
                - 'action': action sequence (relative if use_relative_action=True)
        """
        threadpool_limits(1)
        
        # Get valid start indices
        valid_indices = self._get_valid_indices(self.train_mask)
        start_idx = int(valid_indices[idx])
        
        # Define sequence ranges using slices (zarr compatible)
        obs_slice = slice(start_idx, start_idx + self.n_obs_steps)
        action_slice = slice(start_idx, start_idx + self.horizon)
        
        # Load observations
        obs_dict = {}
        
        # Load images
        for img_key in self.image_keys:
            # Load and convert: (T, H, W, C) -> (T, C, H, W), uint8 -> float32 [0, 1]
            imgs = self.zarr_root[img_key][obs_slice]  # (T, H, W, 3)
            imgs = np.moveaxis(imgs, -1, 1).astype(np.float32) / 255.0  # (T, 3, H, W)
            obs_dict[img_key] = imgs
        
        # Load force data
        force = self.zarr_root[self.force_key][obs_slice].astype(np.float32)  # (T, 6)
        obs_dict[self.force_key] = force
        
        # Load actions
        actions = self.zarr_root[self.action_key][action_slice].astype(np.float32)  # (horizon, action_dim)
        
        # Convert to relative actions if requested
        if self.use_relative_action:
            actions = self._to_relative_action(actions)
        
        # Convert to torch tensors
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(actions)
        }
        
        return torch_data
    
    def _to_relative_action(self, actions: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Convert absolute actions to relative actions using homogeneous transformation.
        
        For 6-DOF pose (x, y, z, rx, ry, rz):
        - Compute relative transform: T_relative = inv(T_reference) @ T_current
        - This gives the current pose in the reference frame's coordinate system
        - Optionally adds noise to reference pose for data augmentation
        
        For gripper: Keep absolute value
        
        Args:
            actions: (horizon, action_dim) absolute actions
                     Expected: [..., x, y, z, rx, ry, rz, gripper]
            add_noise: If True, add Gaussian noise to reference pose for augmentation
        
        Returns:
            (horizon, action_dim) relative actions
        """
        if actions.shape[1] < 6:
            # If less than 6 DOF, fall back to simple subtraction
            return actions - actions[0:1]
        
        relative_actions = np.zeros_like(actions)
        
        # Reference pose (first action in chunk)
        ref_pose = actions[0, :6].copy()  # (x, y, z, rx, ry, rz)
        
        # Add noise to reference pose for data augmentation
        # This makes the model more robust to different starting positions
        if add_noise and hasattr(self, 'use_reference_noise') and self.use_reference_noise:
            ref_pose += np.random.normal(scale=self.reference_noise_scale, size=ref_pose.shape)
        
        # Build reference transformation matrix
        T_ref = self._pose_to_matrix(ref_pose)
        T_ref_inv = np.linalg.inv(T_ref)
        
        # Compute relative transformation for each action
        for i in range(len(actions)):
            current_pose = actions[i, :6]
            T_current = self._pose_to_matrix(current_pose)
            
            # Relative transform: T_rel = T_ref^-1 @ T_current
            T_relative = T_ref_inv @ T_current
            
            # Convert back to pose representation
            relative_actions[i, :6] = self._matrix_to_pose(T_relative)
            
            # Keep gripper absolute
            if actions.shape[1] == 7:
                relative_actions[i, 6] = actions[i, 6]
        
        return relative_actions
    
    def _pose_to_matrix(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert pose (x, y, z, rx, ry, rz) to 4x4 homogeneous transformation matrix.
        
        Args:
            pose: (6,) array [x, y, z, rx, ry, rz]
                  Rotation in euler angles (radians)
        
        Returns:
            T: (4, 4) homogeneous transformation matrix
        """
        x, y, z, rx, ry, rz = pose
        
        # Create rotation matrix from euler angles (XYZ convention)
        rot = R.from_euler('xyz', [rx, ry, rz], degrees=False)
        rot_matrix = rot.as_matrix()
        
        # Build homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = [x, y, z]
        
        return T
    
    def _matrix_to_pose(self, T: np.ndarray) -> np.ndarray:
        """
        Convert 4x4 homogeneous transformation matrix to pose (x, y, z, rx, ry, rz).
        
        Args:
            T: (4, 4) homogeneous transformation matrix
        
        Returns:
            pose: (6,) array [x, y, z, rx, ry, rz]
                  Rotation in euler angles (radians)
        """
        # Extract translation
        x, y, z = T[:3, 3]
        
        # Extract rotation and convert to euler angles
        rot_matrix = T[:3, :3]
        rot = R.from_matrix(rot_matrix)
        rx, ry, rz = rot.as_euler('xyz', degrees=False)
        
        return np.array([x, y, z, rx, ry, rz])
    
    def get_normalizer(self, mode: str = 'limits') -> LinearNormalizer:
        """
        Compute normalization statistics.
        
        Args:
            mode: 'limits' for min-max normalization, 'gaussian' for z-score
        
        Returns:
            LinearNormalizer with statistics for all data fields
        """
        normalizer = LinearNormalizer()
        
        # Sample data for computing statistics (use all training data)
        print("Computing normalization statistics...")
        
        # For images: use standard [0, 1] normalization
        for img_key in self.image_keys:
            normalizer[img_key] = get_image_range_normalizer()
        
        # For force and action: compute from training data
        train_episodes = np.where(self.train_mask)[0]
        
        # Collect force data from training episodes
        force_data_list = []
        for ep_idx in train_episodes:
            start = int(self.episode_starts[ep_idx])
            end = int(self.episode_ends[ep_idx])
            force_data_list.append(self.zarr_root[self.force_key][start:end])
        force_data = np.concatenate(force_data_list, axis=0)
        
        # Collect action data from training episodes
        action_data_list = []
        for ep_idx in train_episodes:
            start = int(self.episode_starts[ep_idx])
            end = int(self.episode_ends[ep_idx])
            actions = self.zarr_root[self.action_key][start:end]
            
            # Convert to relative if needed
            if self.use_relative_action and len(actions) > 0:
                # For computing stats, use relative actions WITHOUT noise
                # Sample multiple chunks to get good coverage
                for i in range(0, len(actions) - self.horizon + 1, self.horizon // 2):
                    chunk = actions[i:i + self.horizon]
                    if len(chunk) == self.horizon:
                        # No noise for computing normalization statistics
                        relative_chunk = self._to_relative_action(chunk, add_noise=False)
                        action_data_list.append(relative_chunk)
            else:
                action_data_list.append(actions)
        
        action_data = np.concatenate(action_data_list, axis=0)
        
        # Create normalizers
        normalizer[self.force_key] = SingleFieldLinearNormalizer.create_fit(
            force_data,
            mode=mode
        )
        normalizer[self.action_key] = SingleFieldLinearNormalizer.create_fit(
            action_data,
            mode=mode
        )
        
        print(f"  Force range: [{force_data.min():.3f}, {force_data.max():.3f}]")
        print(f"  Action range: [{action_data.min():.3f}, {action_data.max():.3f}]")
        
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        """Get all actions as a tensor (for analysis)"""
        # Note: This returns absolute actions, not relative
        return torch.from_numpy(self.zarr_root[self.action_key][:])
    
    def get_episode(self, episode_idx: int) -> Dict[str, np.ndarray]:
        """
        Get all data from a specific episode.
        
        Args:
            episode_idx: Episode index
        
        Returns:
            Dictionary with images, force, and actions for the episode
        """
        start = int(self.episode_starts[episode_idx])
        end = int(self.episode_ends[episode_idx])
        
        episode_data = {}
        
        # Load images
        for img_key in self.image_keys:
            episode_data[img_key] = self.zarr_root[img_key][start:end]
        
        # Load force
        episode_data[self.force_key] = self.zarr_root[self.force_key][start:end]
        
        # Load actions
        episode_data[self.action_key] = self.zarr_root[self.action_key][start:end]
        
        return episode_data


def test_fadp_dataset():
    """Test FADP dataset loading and sampling"""
    import matplotlib.pyplot as plt
    
    # Create dataset
    dataset = FADPDataset(
        dataset_path='data/forceumi1',
        horizon=16,
        n_obs_steps=5,
        n_action_steps=8,
        use_relative_action=True,
        val_ratio=0.1,
    )
    
    # Get normalizer
    normalizer = dataset.get_normalizer()
    
    # Sample a batch
    print("\nSampling a batch...")
    print('len(dataset):', len(dataset))
    index = np.random.randint(0, len(dataset) - 1)
    sample = dataset[index]
    print(f"Observation keys: {sample['obs'].keys()}")
    print(f"Image shape: {sample['obs']['camera_0'].shape}")
    print(f"Force shape: {sample['obs']['force'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print('action:', sample['action'])
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot first image
    img = sample['obs']['camera_0'][0].numpy().transpose(1, 2, 0)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('First Observation Image')
    axes[0, 0].axis('off')
    
    # Plot force
    force = sample['obs']['force'].numpy()
    for i in range(6):
        axes[0, 1].plot(force[:, i], label=f'f{i}')
    axes[0, 1].set_title('Force Observations')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot action trajectory
    action = sample['action'].numpy()
    axes[1, 0].plot(action[:, :3])
    axes[1, 0].set_title('Position Actions (dx, dy, dz)')
    axes[1, 0].legend(['dx', 'dy', 'dz'])
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(action[:, 3:6])
    axes[1, 1].set_title('Rotation Actions (drx, dry, drz)')
    axes[1, 1].legend(['drx', 'dry', 'drz'])
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('fadp_dataset_sample.png')
    print("Saved visualization to fadp_dataset_sample.png")


if __name__ == '__main__':
    test_fadp_dataset()

