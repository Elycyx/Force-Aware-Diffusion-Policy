"""
FADP-UMI Dataset Adapter

这个数据集类继承自UmiDataset，专门用于加载FADP数据集并转换为UMI格式。

关键差异：
- FADP存储的是欧拉角（Euler angles，xyz顺序）
- UMI期望的是rotation vector（axis-angle）

这个类在数据加载时将欧拉角转换为rotation vector，确保与UMI的pose_util兼容。
"""

import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil
import scipy.spatial.transform as st

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.umi_dataset import UmiDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import pose_to_mat, mat_to_pose10d

register_codecs()


class FadpUmiDataset(UmiDataset):
    """
    FADP数据集适配器，继承自UmiDataset
    
    主要功能：
    1. 加载FADP数据集（欧拉角格式）
    2. 转换欧拉角为rotation vector（axis-angle）
    3. 使用UmiDataset的其他所有处理逻辑
    """
    
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None
    ):
        """
        初始化FADP-UMI数据集
        
        在调用父类初始化前，会先转换replay_buffer中的旋转表示
        """
        print("=" * 60)
        print("初始化FadpUmiDataset（FADP→UMI格式转换）")
        print("=" * 60)
        
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        
        # 加载replay buffer（与UmiDataset相同的逻辑）
        if cache_dir is None:
            # load into memory store
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, 
                    store=zarr.MemoryStore()
                )
        else:
            # 使用缓存（与UmiDataset相同）
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        # ============ 关键步骤：转换FADP的欧拉角为UMI的rotation vector ============
        print("\n开始转换旋转表示：欧拉角 → rotation vector (axis-angle)")
        
        # 转换观测数据中的旋转
        for key in replay_buffer.keys():
            if 'eef_rot_axis_angle' in key and not key.endswith('_wrt_start'):
                # FADP中这个key存储的是欧拉角，需要转换为rotation vector
                euler_data = replay_buffer[key][:]  # (N, 3) 欧拉角 (roll, pitch, yaw)
                
                print(f"\n转换 {key}:")
                print(f"  原始数据shape: {euler_data.shape}")
                print(f"  原始格式: 欧拉角 (Euler angles, xyz order)")
                print(f"  目标格式: rotation vector (axis-angle)")
                
                # 转换：欧拉角 → rotation vector
                # 使用scipy的Rotation类进行转换
                rotvec_data = np.zeros_like(euler_data)
                for i in range(len(euler_data)):
                    # 从欧拉角创建Rotation对象
                    rot = st.Rotation.from_euler('xyz', euler_data[i], degrees=False)
                    # 转换为rotation vector (axis-angle)
                    rotvec_data[i] = rot.as_rotvec()
                
                # 替换replay_buffer中的数据
                replay_buffer.data[key][:] = rotvec_data
                
                print(f"  ✓ 转换完成")
                print(f"    示例值（前3个）:")
                print(f"      欧拉角: {euler_data[0]}")
                print(f"      rotvec: {rotvec_data[0]}")
        
        # 转换action中的旋转（如果action存在）
        if 'action' in replay_buffer.keys():
            action_data = replay_buffer['action'][:]  # (N, 7) - [x, y, z, rx, ry, rz, gripper]
            
            print(f"\n转换 action 中的旋转:")
            print(f"  原始action shape: {action_data.shape}")
            
            # 假设是单机器人，action是7维：[pos(3), euler(3), gripper(1)]
            # 需要转换中间的3维欧拉角为rotation vector
            num_robots = action_data.shape[-1] // 7
            
            for robot_id in range(num_robots):
                start_idx = robot_id * 7
                euler_action = action_data[:, start_idx + 3: start_idx + 6]  # (N, 3)
                
                rotvec_action = np.zeros_like(euler_action)
                for i in range(len(euler_action)):
                    rot = st.Rotation.from_euler('xyz', euler_action[i], degrees=False)
                    rotvec_action[i] = rot.as_rotvec()
                
                # 替换action中的旋转部分
                action_data[:, start_idx + 3: start_idx + 6] = rotvec_action
            
            replay_buffer.data['action'][:] = action_data
            print(f"  ✓ Action转换完成")
        
        print("\n" + "=" * 60)
        print("旋转格式转换完成！")
        print("=" * 60)
        
        # 现在replay_buffer中的数据已经是rotation vector格式
        # 可以安全地使用UmiDataset的其余逻辑
        
        # 调用父类初始化之前，需要先设置这些属性
        # 因为父类会直接使用replay_buffer
        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.action_padding = action_padding
        self.temporally_independent_normalization = temporally_independent_normalization
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.threadpool_limits_is_applied = False
        
        # 解析shape_meta（与UmiDataset相同）
        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1

            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
    
        for key in replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                if query_key in shape_meta['obs']:
                    key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                    key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                    key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )
        
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.sampler = sampler
        
        print(f"\n数据集初始化完成:")
        print(f"  训练样本数: {len(self.sampler)}")
        print(f"  验证样本比例: {val_ratio}")
        print(f"  机器人数量: {self.num_robot}")
        print()
    
    # 其他方法直接继承自UmiDataset，无需重写
    # get_validation_dataset, get_normalizer, __len__, __getitem__ 都使用父类实现
