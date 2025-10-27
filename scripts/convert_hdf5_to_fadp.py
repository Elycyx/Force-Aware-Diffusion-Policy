#!/usr/bin/env python3
"""
Convert HDF5 episode format to FADP zarr format.

Usage:
    python scripts/convert_hdf5_to_fadp.py \
        --input data/session_* \
        --output data/fadp_dataset \
        --image-size 240 320 \
        --max-episodes 100

Author: FADP Team
"""

import argparse
import h5py
import zarr
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional
import cv2


class HDF5ToFADPConverter:
    """Convert HDF5 episode data to FADP zarr format"""
    
    def __init__(
        self,
        input_pattern: str,
        output_dir: str,
        image_size: Tuple[int, int] = (240, 320),
        max_episodes: Optional[int] = None,
        validate: bool = True,
    ):
        """
        Args:
            input_pattern: Glob pattern for input sessions (e.g., "data/session_*")
            output_dir: Output directory for FADP dataset
            image_size: Target image size (height, width)
            max_episodes: Maximum number of episodes to process (None = all)
            validate: Whether to validate data during conversion
        """
        self.input_pattern = input_pattern
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.max_episodes = max_episodes
        self.validate = validate
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def find_all_episodes(self) -> List[Path]:
        """Find all episode HDF5 files from session directories"""
        from glob import glob
        
        episodes = []
        session_dirs = sorted(glob(self.input_pattern))
        
        print(f"Found {len(session_dirs)} session directories")
        
        for session_dir in session_dirs:
            session_path = Path(session_dir)
            episode_files = sorted(session_path.glob("episode*.hdf5"))
            episodes.extend(episode_files)
            print(f"  {session_path.name}: {len(episode_files)} episodes")
        
        if self.max_episodes:
            episodes = episodes[:self.max_episodes]
            print(f"\nLimited to first {self.max_episodes} episodes")
        
        print(f"\nTotal episodes to process: {len(episodes)}")
        return episodes
    
    def load_episode(self, episode_path: Path) -> dict:
        """
        Load data from a single HDF5 episode file.
        
        Returns:
            Dictionary containing:
                - image: (N, H, W, 3) uint8
                - force: (N, 6) float32
                - action: (N, 7) float32
                - state: (N, 7) float32 (optional, not used in FADP)
                - metadata: dict
        """
        with h5py.File(episode_path, 'r') as f:
            # Load required data
            images = f['image'][:]  # (N, H, W, 3)
            force = f['force'][:]   # (N, 6)
            action = f['action'][:] # (N, 7)
            
            # Resize images if needed
            if images.shape[1:3] != self.image_size:
                print(f"  Resizing images from {images.shape[1:3]} to {self.image_size}")
                resized_images = []
                for img in images:
                    resized = cv2.resize(img, (self.image_size[1], self.image_size[0]))
                    resized_images.append(resized)
                images = np.array(resized_images, dtype=np.uint8)
            
            # Load metadata if available
            metadata = {}
            if 'metadata' in f.attrs:
                metadata = dict(f.attrs['metadata'])
            
            # Optional: load state for validation
            state = f['state'][:] if 'state' in f else None
            
            data = {
                'image': images,
                'force': force,
                'action': action,
                'state': state,
                'metadata': metadata,
            }
            
            # Validate
            if self.validate:
                self._validate_episode_data(data, episode_path)
            
            return data
    
    def _validate_episode_data(self, data: dict, episode_path: Path):
        """Validate episode data for consistency"""
        n_frames = len(data['image'])
        
        # Check all arrays have same length
        assert len(data['force']) == n_frames, \
            f"{episode_path}: Force length {len(data['force'])} != image length {n_frames}"
        assert len(data['action']) == n_frames, \
            f"{episode_path}: Action length {len(data['action'])} != image length {n_frames}"
        
        # Check shapes
        assert data['image'].shape[1:] == (*self.image_size, 3), \
            f"{episode_path}: Image shape {data['image'].shape[1:]} != expected {(*self.image_size, 3)}"
        assert data['force'].shape[1] == 6, \
            f"{episode_path}: Force shape {data['force'].shape[1]} != 6"
        assert data['action'].shape[1] == 7, \
            f"{episode_path}: Action shape {data['action'].shape[1]} != 7"
        
        # Check data types
        assert data['image'].dtype == np.uint8, \
            f"{episode_path}: Image dtype {data['image'].dtype} != uint8"
        assert data['force'].dtype == np.float32, \
            f"{episode_path}: Force dtype {data['force'].dtype} != float32"
        assert data['action'].dtype == np.float32, \
            f"{episode_path}: Action dtype {data['action'].dtype} != float32"
        
        # Check value ranges
        assert data['image'].min() >= 0 and data['image'].max() <= 255, \
            f"{episode_path}: Image values out of range [0, 255]"
        
        # Warn about extreme force values
        force_max = np.abs(data['force']).max()
        if force_max > 100:
            print(f"  ⚠️  Warning: Large force value {force_max:.1f} in {episode_path.name}")
        
        # Warn about extreme action values
        action_max = np.abs(data['action'][:, :6]).max()  # Position/rotation deltas
        if action_max > 0.5:
            print(f"  ⚠️  Warning: Large action delta {action_max:.3f} in {episode_path.name}")
    
    def convert(self):
        """Main conversion function with streaming to minimize memory usage"""
        print("="*60)
        print("HDF5 to FADP Dataset Converter (Memory-Efficient)")
        print("="*60)
        
        # Find all episodes
        episode_files = self.find_all_episodes()
        
        if len(episode_files) == 0:
            print("No episodes found! Check your input pattern.")
            return
        
        # First pass: Calculate total size and validate
        print("\nPass 1: Analyzing episodes...")
        total_frames = 0
        episode_lengths = []
        valid_episodes = []
        
        for episode_path in tqdm(episode_files, desc="Analyzing"):
            try:
                with h5py.File(episode_path, 'r') as f:
                    n_frames = len(f['image'])
                    episode_lengths.append(n_frames)
                    total_frames += n_frames
                    valid_episodes.append(episode_path)
            except Exception as e:
                print(f"\n❌ Error reading {episode_path.name}: {e}")
                continue
        
        if len(valid_episodes) == 0:
            print("\n❌ No valid episodes found!")
            return
        
        print(f"\nDataset info:")
        print(f"  Valid episodes: {len(valid_episodes)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Estimated memory: {self._estimate_memory(total_frames)} MB")
        
        # Prepare zarr store
        zarr_path = self.output_dir / 'data.zarr'
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=True)
        
        # Pre-allocate zarr arrays (memory-mapped, not in RAM)
        print("\nCreating zarr arrays...")
        camera_array = root.create_dataset(
            'camera_0',
            shape=(total_frames, self.image_size[0], self.image_size[1], 3),
            dtype='uint8',
            chunks=(10, self.image_size[0], self.image_size[1], 3),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )
        
        force_array = root.create_dataset(
            'force',
            shape=(total_frames, 6),
            dtype='float32',
            chunks=(100, 6),
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )
        
        action_array = root.create_dataset(
            'action',
            shape=(total_frames, 7),
            dtype='float32',
            chunks=(100, 7),
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )
        
        # Second pass: Stream data to zarr
        print("\nPass 2: Writing data to zarr...")
        episode_ends = []
        current_idx = 0
        
        # Collect statistics for final report
        force_stats = {'sum': np.zeros(6), 'sum_sq': np.zeros(6), 'min': np.full(6, np.inf), 'max': np.full(6, -np.inf)}
        action_stats = {'sum': np.zeros(7), 'sum_sq': np.zeros(7), 'min': np.full(7, np.inf), 'max': np.full(7, -np.inf)}
        
        for episode_path in tqdm(valid_episodes, desc="Writing"):
            try:
                # Load ONE episode at a time
                episode_data = self.load_episode(episode_path)
                
                n_frames = len(episode_data['image'])
                end_idx = current_idx + n_frames
                
                # Write directly to zarr (memory-mapped)
                camera_array[current_idx:end_idx] = episode_data['image']
                force_array[current_idx:end_idx] = episode_data['force']
                action_array[current_idx:end_idx] = episode_data['action']
                
                # Update statistics incrementally
                self._update_stats(force_stats, episode_data['force'])
                self._update_stats(action_stats, episode_data['action'])
                
                # Update indices
                current_idx = end_idx
                episode_ends.append(end_idx)
                
                # Free memory immediately
                del episode_data
                
            except Exception as e:
                print(f"\n❌ Error processing {episode_path.name}: {e}")
                print(f"   Skipping this episode...")
                continue
        
        # Save episode metadata
        print("\nSaving metadata...")
        meta_dir = self.output_dir / 'meta'
        meta_dir.mkdir(exist_ok=True)
        
        with open(meta_dir / 'episode_ends.json', 'w') as f:
            json.dump(episode_ends, f, indent=2)
        
        # Save conversion info
        conversion_info = {
            'source_pattern': self.input_pattern,
            'num_episodes': len(episode_ends),
            'total_frames': total_frames,
            'image_size': list(self.image_size),
            'episode_ends': episode_ends,
        }
        
        with open(self.output_dir / 'conversion_info.json', 'w') as f:
            json.dump(conversion_info, f, indent=2)
        
        print(f"\n✅ Conversion complete!")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Zarr store: {zarr_path}")
        
        # Print statistics (using collected stats, not loading data)
        self._print_statistics_from_stats(force_stats, action_stats, episode_ends, total_frames)
    
    def _estimate_memory(self, total_frames: int) -> int:
        """Estimate peak memory usage in MB"""
        # Image: H*W*3 bytes per frame
        image_mem = total_frames * self.image_size[0] * self.image_size[1] * 3
        # Force: 6 * 4 bytes per frame
        force_mem = total_frames * 6 * 4
        # Action: 7 * 4 bytes per frame
        action_mem = total_frames * 7 * 4
        
        total_mb = (image_mem + force_mem + action_mem) / (1024 ** 2)
        return int(total_mb)
    
    def _update_stats(self, stats: dict, data: np.ndarray):
        """Update statistics incrementally without storing all data"""
        stats['sum'] += data.sum(axis=0)
        stats['sum_sq'] += (data ** 2).sum(axis=0)
        stats['min'] = np.minimum(stats['min'], data.min(axis=0))
        stats['max'] = np.maximum(stats['max'], data.max(axis=0))
    
    def _print_statistics_from_stats(self, force_stats: dict, action_stats: dict, episode_ends: list, total_frames: int):
        """Print dataset statistics from incrementally collected stats (memory-efficient)"""
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        
        # Episode lengths
        episode_lengths = []
        for i, end in enumerate(episode_ends):
            start = 0 if i == 0 else episode_ends[i-1]
            length = end - start
            episode_lengths.append(length)
        
        print(f"\nEpisodes:")
        print(f"  Count: {len(episode_ends)}")
        print(f"  Length: min={min(episode_lengths)}, max={max(episode_lengths)}, mean={np.mean(episode_lengths):.1f}")
        
        # Force statistics (from incremental stats)
        print(f"\nForce/Torque (N, N·m):")
        labels = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
        for i, label in enumerate(labels):
            mean = force_stats['sum'][i] / total_frames
            # std = sqrt(E[X^2] - E[X]^2)
            std = np.sqrt(force_stats['sum_sq'][i] / total_frames - mean**2)
            print(f"  {label}: mean={mean:+.3f}, std={std:.3f}, range=[{force_stats['min'][i]:+.3f}, {force_stats['max'][i]:+.3f}]")
        
        # Action statistics (from incremental stats)
        print(f"\nActions:")
        labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
        for i, label in enumerate(labels):
            mean = action_stats['sum'][i] / total_frames
            std = np.sqrt(action_stats['sum_sq'][i] / total_frames - mean**2)
            print(f"  {label}: mean={mean:+.4f}, std={std:.4f}, range=[{action_stats['min'][i]:+.4f}, {action_stats['max'][i]:+.4f}]")
        
        # Storage size
        zarr_path = self.output_dir / 'data.zarr'
        total_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
        print(f"\nStorage:")
        print(f"  Total size: {total_size / 1024**2:.1f} MB")
        print(f"  Per frame: {total_size / total_frames / 1024:.1f} KB")
        print(f"\nMemory-efficient conversion complete! 🎉")
    
    def _print_statistics(self, images, forces, actions, episode_ends):
        """Legacy function - kept for compatibility but not used in streaming mode"""
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        
        # Episode lengths
        episode_lengths = []
        for i, end in enumerate(episode_ends):
            start = 0 if i == 0 else episode_ends[i-1]
            length = end - start
            episode_lengths.append(length)
        
        print(f"\nEpisodes:")
        print(f"  Count: {len(episode_ends)}")
        print(f"  Length: min={min(episode_lengths)}, max={max(episode_lengths)}, mean={np.mean(episode_lengths):.1f}")
        
        # Force statistics
        print(f"\nForce/Torque (N, N·m):")
        labels = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
        for i, label in enumerate(labels):
            data = forces[:, i]
            print(f"  {label}: mean={data.mean():+.3f}, std={data.std():.3f}, range=[{data.min():+.3f}, {data.max():+.3f}]")
        
        # Action statistics
        print(f"\nActions:")
        labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
        for i, label in enumerate(labels):
            data = actions[:, i]
            print(f"  {label}: mean={data.mean():+.4f}, std={data.std():.4f}, range=[{data.min():+.4f}, {data.max():+.4f}]")
        
        # Storage size
        zarr_path = self.output_dir / 'data.zarr'
        total_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
        print(f"\nStorage:")
        print(f"  Total size: {total_size / 1024**2:.1f} MB")
        print(f"  Per frame: {total_size / len(images) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 episodes to FADP zarr format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all sessions
  python scripts/convert_hdf5_to_fadp.py \\
      --input "data/session_*" \\
      --output data/fadp_dataset

  # Convert specific sessions with custom image size
  python scripts/convert_hdf5_to_fadp.py \\
      --input "data/session_20250118_*" \\
      --output data/fadp_dataset \\
      --image-size 96 96

  # Convert limited number of episodes
  python scripts/convert_hdf5_to_fadp.py \\
      --input "data/session_*" \\
      --output data/fadp_dataset \\
      --max-episodes 50
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input pattern for session directories (e.g., "data/session_*")'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for FADP dataset'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=[240, 320],
        metavar=('HEIGHT', 'WIDTH'),
        help='Target image size (default: 240 320)'
    )
    
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=None,
        help='Maximum number of episodes to process (default: all)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Disable data validation during conversion'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = HDF5ToFADPConverter(
        input_pattern=args.input,
        output_dir=args.output,
        image_size=tuple(args.image_size),
        max_episodes=args.max_episodes,
        validate=not args.no_validate,
    )
    
    # Run conversion
    try:
        converter.convert()
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()

