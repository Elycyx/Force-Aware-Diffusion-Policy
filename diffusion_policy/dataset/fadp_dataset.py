import copy
from typing import Dict, Optional, Union, List

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from filelock import FileLock
import shutil

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from scipy.spatial.transform import Rotation as R

register_codecs()


def pose_to_mat(pose):
    """
    将pose表示转换为4x4齐次变换矩阵
    
    Args:
        pose: (..., 6) numpy array [x, y, z, rx, ry, rz] 
              其中 (rx, ry, rz) 是欧拉角表示 (roll, pitch, yaw)，单位：弧度
    
    Returns:
        mat: (..., 4, 4) 齐次变换矩阵
    
    原理：
        齐次变换矩阵的形式为:
        [[R11, R12, R13, x],
         [R21, R22, R23, y],
         [R31, R32, R33, z],
         [0,   0,   0,   1]]
        其中R是3x3旋转矩阵，(x,y,z)是平移向量
    """
    assert pose.shape[-1] == 6, f"Pose应该是6维，实际为{pose.shape[-1]}"
    
    # 处理批次维度
    original_shape = pose.shape[:-1]
    pose = pose.reshape(-1, 6)
    batch_size = pose.shape[0]
    
    # 分离位置和旋转
    pos = pose[:, :3]  # (N, 3)
    euler = pose[:, 3:6]  # (N, 3) 欧拉角 (roll, pitch, yaw)
    
    # 初始化变换矩阵
    mat = np.zeros((batch_size, 4, 4), dtype=np.float32)
    
    # 设置旋转部分 - 使用欧拉角 (xyz顺序，即roll, pitch, yaw)
    for i in range(batch_size):
        rot = R.from_euler('xyz', euler[i], degrees=False)
        mat[i, :3, :3] = rot.as_matrix()
    
    # 设置平移部分
    mat[:, :3, 3] = pos
    
    # 设置齐次坐标
    mat[:, 3, 3] = 1.0
    
    # 恢复原始形状
    mat = mat.reshape(original_shape + (4, 4))
    
    return mat


class FadpDataset(BaseDataset):
    """
    简化的数据集类，只输出RGB图像和action（不输出state观测）
    
    数据格式:
    - camera0_rgb: RGB图像 (T, H, W, 3)
    - action: 动作 (T, 7) - [x, y, z, rx, ry, rz, gripper] (相对pose，旋转使用欧拉角 roll, pitch, yaw)
    
    注意：虽然不输出state观测，但会使用state数据来计算相对action
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
        max_duration: Optional[float]=None,
        pose_noise_scale: Union[float, List[float], np.ndarray]=0.0
    ):
        """
        初始化FADP数据集
        
        Args:
            shape_meta: 数据形状元信息
            dataset_path: zarr数据集路径
            cache_dir: 缓存目录（可选）
            pose_repr: pose表示配置
            action_padding: 是否使用action padding
            temporally_independent_normalization: 是否使用时间独立归一化
            repeat_frame_prob: 重复帧的概率
            seed: 随机种子
            val_ratio: 验证集比例
            max_duration: 最大持续时间
            pose_noise_scale: 位姿噪声标准差，可以是：
                - 单个float值：所有6个维度使用相同标准差
                - 6个值的列表/数组：[x, y, z, rx, ry, rz] 分别对应各维度的标准差
                - 0表示不添加噪声
        """
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        if cache_dir is None:
            # 加载到内存
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, 
                    store=zarr.MemoryStore()
                )
        else:
            # 使用LMDB缓存
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            print('获取缓存锁...')
            with FileLock(lock_path):
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"复制数据到缓存: {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("缓存写入完成!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # 打开只读LMDB
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        # 解析shape_meta，提取RGB和lowdim信息
        # 注意：我们需要读取state数据用于计算相对action，但不作为观测输出
        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()  # 用于sampler读取state数据
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
            
            key_horizon[key] = attr['horizon']
            key_latency_steps[key] = attr['latency_steps']
            key_down_sample_steps[key] = attr['down_sample_steps']
        
        # action配置
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']
        
        # 划分训练/验证集
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        
        # sampler需要的lowdim_keys（用于读取state，但不作为观测输出）
        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
        
        # 创建序列采样器（需要读取state用于action处理）
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
        
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        
        # 处理pose_noise_scale：统一转换为6个值的数组
        # 支持 OmegaConf ListConfig
        if isinstance(pose_noise_scale, (list, tuple, np.ndarray)):
            # 如果是列表/数组，确保有6个值
            pose_noise_scale = np.array(pose_noise_scale, dtype=np.float32)
            if len(pose_noise_scale) != 6:
                raise ValueError(f"pose_noise_scale列表长度必须为6，实际为{len(pose_noise_scale)}")
            self.pose_noise_scale = pose_noise_scale
        else:
            # 尝试转换为列表（处理 OmegaConf ListConfig）
            try:
                pose_noise_list = list(pose_noise_scale)
                pose_noise_scale = np.array(pose_noise_list, dtype=np.float32)
                if len(pose_noise_scale) != 6:
                    raise ValueError(f"pose_noise_scale列表长度必须为6，实际为{len(pose_noise_scale)}")
                self.pose_noise_scale = pose_noise_scale
            except (TypeError, ValueError) as e:
                # 如果是单个值，扩展为6个相同的值
                try:
                    scale_value = float(pose_noise_scale)
                    self.pose_noise_scale = np.array([scale_value] * 6, dtype=np.float32)
                except (TypeError, ValueError):
                    raise ValueError(f"无法将 pose_noise_scale 转换为有效值: {pose_noise_scale}") from e
        
        self.threadpool_limits_is_applied = False
    
    def get_validation_dataset(self):
        """创建验证集（不使用噪声）"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~self.val_mask
        # 验证集不使用噪声（设置为全0数组）
        val_set.pose_noise_scale = np.zeros(6, dtype=np.float32)
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        计算归一化参数
        
        对于action (可能是7维或13维):
        - 7维格式: [x, y, z, rx, ry, rz, gripper]
          - 前3维 (x,y,z): 使用range normalization
          - 中3维 (rx,ry,rz): 使用identity normalization (欧拉角 roll, pitch, yaw)
          - 最后1维 (gripper): 使用range normalization
        
        - 13维格式: [x, y, z, rx, ry, rz, gripper, fx, fy, fz, mx, my, mz]
          - 前3维 (x,y,z): 使用range normalization
          - 中3维 (rx,ry,rz): 使用identity normalization (欧拉角 roll, pitch, yaw)
          - 第7维 (gripper): 使用range normalization
          - 后6维 (fx, fy, fz, mx, my, mz): 使用range normalization
        
        对于force观测 (6维: fx, fy, fz, mx, my, mz):
        - 使用range normalization（如果存在force数据）
        
        对于RGB图像: 使用[0,1]归一化
        """
        normalizer = LinearNormalizer()
        
        # 收集action和force数据用于计算统计信息
        action_data = list()
        force_data_dict = {key: list() for key in self.lowdim_keys if 'force' in key.lower()}
        
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=8,
        )
        for batch in tqdm(dataloader, desc='计算归一化参数'):
            action_data.append(copy.deepcopy(batch['action']))
            # 收集force数据（如果存在）
            if 'obs' in batch:
                for key in force_data_dict.keys():
                    if key in batch['obs']:
                        force_data_dict[key].append(copy.deepcopy(batch['obs'][key]))
        self.sampler.ignore_rgb(False)
        
        # 合并所有action数据
        action_data = np.concatenate(action_data)
        assert action_data.shape[0] == len(self.sampler)
        assert len(action_data.shape) == 3  # (B, T, D)
        B, T, D = action_data.shape
        
        if not self.temporally_independent_normalization:
            action_data = action_data.reshape(B*T, D)
        
        # 为action的不同部分创建不同的归一化器
        # 检测action维度: 7维或13维
        action_dim_total = action_data.shape[-1]
        
        if action_dim_total % 13 == 0:
            # 13维格式: 7维action + 6维force
            dim_a = 13
            has_force_in_action = True
        elif action_dim_total % 7 == 0:
            # 7维格式: 仅action
            dim_a = 7
            has_force_in_action = False
        else:
            raise ValueError(f"Unexpected action dimension: {action_dim_total}")
        
        n_robots = action_dim_total // dim_a
        action_normalizers = list()
        
        for i in range(n_robots):
            start_idx = i * dim_a
            
            # 位置 (3维) - range normalization
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(action_data[..., start_idx: start_idx + 3])
                )
            )
            
            # 旋转 (3维 欧拉角 roll, pitch, yaw) - identity normalization
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(action_data[..., start_idx + 3: start_idx + 6])
                )
            )
            
            # 夹爪 (1维) - range normalization
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(action_data[..., start_idx + 6: start_idx + 7])
                )
            )
            
            # 如果包含force (6维) - range normalization
            if has_force_in_action:
                action_normalizers.append(
                    get_range_normalizer_from_stat(
                        array_to_stats(action_data[..., start_idx + 7: start_idx + 13])
                    )
                )
        
        normalizer['action'] = concatenate_normalizer(action_normalizers)
        
        # Force归一化（如果存在）
        for key, data_list in force_data_dict.items():
            if len(data_list) > 0:
                force_data = np.concatenate(data_list)
                assert force_data.shape[0] == len(self.sampler)
                assert len(force_data.shape) == 3  # (B, T, 6)
                
                if not self.temporally_independent_normalization:
                    B, T, D = force_data.shape
                    force_data = force_data.reshape(B*T, D)
                
                # Force使用range normalization
                normalizer[key] = get_range_normalizer_from_stat(
                    array_to_stats(force_data)
                )
        
        # 图像归一化（简单的[0,1]归一化）
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        
        return normalizer
    
    def __len__(self):
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            dict with keys:
                - 'obs': dict with RGB images only (不包含state)
                - 'action': action tensor (T, 7) - relative action [x,y,z,rx,ry,rz,gripper]
        """
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        
        # 从sampler获取数据（包含RGB、state和action）
        data = self.sampler.sample_sequence(idx)
        
        # 处理RGB图像（输出）
        obs_dict = dict()
        for key in self.rgb_keys:
            if key not in data:
                continue
            # 将通道从最后移到第二维: (T,H,W,C) -> (T,C,H,W)
            # 转换uint8到float32并归一化到[0,1]
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            del data[key]
        
        # 读取state数据
        # - 包含'eef'或'gripper'的key: 用于计算相对action，但不输出
        # - 包含'force'的key: 
        #   * 历史部分作为观测输出到obs_dict
        #   * 需要从replay buffer重新读取与action时间范围对齐的未来force
        state_dict = dict()
        future_force_data = None  # 用于存储future force数据（与action时间对齐）
        
        for key in self.sampler_lowdim_keys:
            if key in data:
                data_value = data[key].astype(np.float32)
                
                # Force数据特殊处理
                if 'force' in key.lower():
                    # 观测force（历史，用于encoder）
                    obs_dict[key] = data_value  # (T_force_obs, 6) - 例如6步
                    
                    # ⚠️ 关键修复：从replay buffer重新读取与action时间范围对齐的force
                    # data_value只有force_obs_horizon步（如6步）
                    # 我们需要action_horizon步（如16步）的future force
                    # 
                    # 获取当前样本的时间信息（来自sampler）
                    current_idx, start_idx, end_idx, before_first_grasp = self.sampler.indices[idx]
                    
                    # 计算action的时间范围（与sampler中action采样逻辑一致）
                    action_horizon = self.key_horizon['action']
                    action_down_sample = self.key_down_sample_steps['action']
                    action_latency = self.key_latency_steps['action']
                    
                    # action的起始索引（与sampler逻辑一致）
                    action_start_idx = current_idx - action_latency
                    action_end_idx = action_start_idx + (action_horizon - 1) * action_down_sample + 1
                    
                    # 从sampler的replay_buffer读取force（与其他lowdim数据一致）
                    if key in self.sampler.replay_buffer:
                        force_arr = self.sampler.replay_buffer[key]
                        
                        # 采样force序列（使用与action相同的逻辑）
                        force_indices = np.arange(action_start_idx, min(action_end_idx, end_idx), action_down_sample)
                        future_force_data = force_arr[force_indices].astype(np.float32)
                        
                        # 如果长度不足action_horizon，使用padding
                        if len(future_force_data) < action_horizon:
                            if self.action_padding and len(future_force_data) > 0:
                                # 用最后一个值填充
                                pad_length = action_horizon - len(future_force_data)
                                padding = np.tile(future_force_data[-1:], (pad_length, 1))
                                future_force_data = np.concatenate([future_force_data, padding], axis=0)
                            else:
                                # 如果没有padding，截断到实际长度
                                pass
                    
                else:
                    # State数据（eef_pos, eef_rot, gripper）用于action计算
                    state_dict[key] = data_value
                
                del data[key]
        
        # ============ 处理action：转换为相对表示 ============
        # 
        # Relative Action的计算过程：
        # 
        # 1. 输入数据：
        #    - state: 当前机器人末端的位姿 [x, y, z, rx, ry, rz]
        #    - action: 目标位姿（绝对坐标系） [x', y', z', rx', ry', rz']
        # 
        # 2. 转换为变换矩阵：
        #    - current_pose_mat (T_current): 当前位姿的4x4齐次变换矩阵
        #    - action_pose_mat (T_target): 目标位姿的4x4齐次变换矩阵
        # 
        # 3. 计算相对变换：
        #    使用convert_pose_mat_rep函数，计算从当前位姿到目标位姿的相对变换
        #    T_relative = T_current^(-1) * T_target
        #    这个相对变换表示：如果从当前位姿出发，需要进行怎样的变换才能到达目标位姿
        # 
        # 4. 提取相对位姿：
        #    从T_relative中提取平移和旋转，得到相对action
        #    - 相对位置: T_relative的平移部分
        #    - 相对旋转: T_relative的旋转部分（转换为欧拉角 roll, pitch, yaw）
        # 
        # 5. 数据增强（可选）：
        #    在当前位姿上添加噪声，增强模型的鲁棒性
        # 
        # 6. 输出：
        #    relative_action = [dx, dy, dz, drx, dry, drz, gripper]
        #    其中(dx, dy, dz, drx, dry, drz)表示从当前位姿到目标位姿的相对变换
        # 
        actions = list()
        for robot_id in range(self.num_robot):
            # 步骤1: 构建当前位姿矩阵 (T, 4, 4)
            current_pose = np.concatenate([
                state_dict[f'robot{robot_id}_eef_pos'],
                state_dict[f'robot{robot_id}_eef_rot_axis_angle']  # 实际是欧拉角 (roll, pitch, yaw)
            ], axis=-1)  # (T, 6) - [x, y, z, roll, pitch, yaw]
            
            # 数据增强: 在当前位姿上添加噪声（仅训练集）
            # 参考umi_dataset.py中的做法，对位姿的每个维度分别添加高斯噪声
            # self.pose_noise_scale 是6个值的数组：[x, y, z, roll, pitch, yaw] 的标准差
            if np.any(self.pose_noise_scale > 0):
                # 对每个维度分别生成噪声
                noise = np.zeros_like(current_pose)
                for dim in range(6):
                    if self.pose_noise_scale[dim] > 0:
                        noise[:, dim] = np.random.normal(
                            scale=self.pose_noise_scale[dim],
                            size=current_pose.shape[0]
                        )
                current_pose = current_pose + noise
            
            pose_mat = pose_to_mat(current_pose)  # (T, 4, 4)
            
            # 步骤2: 构建目标位姿矩阵 (T, 4, 4)
            target_pose = data['action'][..., 7 * robot_id: 7 * robot_id + 6]  # (T, 6) - [x, y, z, roll, pitch, yaw]
            action_mat = pose_to_mat(target_pose)  # (T, 4, 4)
            
            # 步骤3: 计算相对变换
            # 使用最后一个时刻的位姿作为参考基准
            # 相对变换 = 当前位姿^(-1) * 目标位姿
            action_pose_mat = convert_pose_mat_rep(
                action_mat,  # 目标位姿
                base_pose_mat=pose_mat[-1],  # 参考位姿（最后一个时刻）
                pose_rep=self.obs_pose_repr,  # 'relative'
                backward=False
            )  # (T, 4, 4)
            
            # 步骤4: 从相对变换矩阵提取位置和旋转
            T = action_pose_mat.shape[0]
            action_pos = action_pose_mat[:, :3, 3]  # (T, 3) 相对位置
            
            # 提取旋转矩阵并转换为欧拉角 (roll, pitch, yaw)
            action_rot_list = []
            for t in range(T):
                rot_mat = action_pose_mat[t, :3, :3]
                rot = R.from_matrix(rot_mat)
                euler = rot.as_euler('xyz', degrees=False)  # 欧拉角表示 (roll, pitch, yaw)，单位：弧度
                action_rot_list.append(euler)
            action_rot = np.stack(action_rot_list, axis=0)  # (T, 3)
            
            # 提取gripper（保持不变）
            action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
            
            # 步骤5: 组合相对action
            # 3 pos + 3 rot + 1 gripper = 7维
            actions.append(np.concatenate([action_pos, action_rot, action_gripper], axis=-1))
        
        # 合并所有机器人的action
        action = np.concatenate(actions, axis=-1)  # (T_action, 7*num_robot)
        
        # ============ 添加未来force到action ============
        # 
        # 预测相对力（增量）：与action的相对表示保持一致
        # 
        # 理由：
        # 1. 与action的相对表示一致，训练更稳定
        # 2. 力的变化范围通常比绝对值小，更容易学习和归一化
        # 3. 更关注力的变化趋势（接触检测、力控制等）
        # 4. 减少对初始力标定误差的依赖
        # 
        # 数据维度说明：
        # - action: (T_action, 7) 其中 T_action = action_horizon
        # - obs_dict['force']: (T_force_obs, 6) 其中 T_force_obs = force_obs_horizon (历史)
        # - future_force_data: sampler返回的完整force数据
        # 
        if 'force' in self.lowdim_keys and future_force_data is not None:
            # future_force_data是sampler返回的完整数据
            # 它的长度应该与action的长度一致（都经过相同的采样）
            T_action = action.shape[0]
            
            # 获取当前观测的最后一个force值作为参考（相对力的基准）
            if 'force' in obs_dict and obs_dict['force'].shape[0] > 0:
                current_force = obs_dict['force'][-1]  # (6,) 当前时刻的force（参考值）
            else:
                # 如果没有观测force，使用future_force_data的第一个值作为参考
                current_force = future_force_data[0]  # (6,)
            
            # 处理future_force_data的长度
            if future_force_data.shape[0] == T_action:
                future_force_abs = future_force_data  # (T_action, 6)
            elif future_force_data.shape[0] > T_action:
                future_force_abs = future_force_data[:T_action]  # (T_action, 6)
            else:
                # future_force_data更短，需要填充
                # 这种情况不应该发生，但以防万一
                pad_length = T_action - future_force_data.shape[0]
                # 使用最后一个值填充
                last_force = future_force_data[-1]  # (6,)
                padding = np.tile(last_force, (pad_length, 1))  # (pad_length, 6)
                future_force_abs = np.concatenate([future_force_data, padding], axis=0)  # (T_action, 6)
            
            # 计算相对力（增量）：future_force - current_force
            # 这样模型预测的是相对于当前力的变化量
            future_force_delta = future_force_abs - current_force  # (T_action, 6)
            
            # 拼接action和相对force
            action = np.concatenate([action, future_force_delta], axis=-1)  # (T_action, 13)
        elif 'force' in self.lowdim_keys:
            # Fallback: 如果没有future_force_data，使用零增量
            # 表示力不变（相对增量为0）
            T_action = action.shape[0]
            future_force_delta = np.zeros((T_action, 6), dtype=np.float32)
            action = np.concatenate([action, future_force_delta], axis=-1)  # (T_action, 13)
        
        # 转换为PyTorch tensors（只输出RGB观测和action）
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action.astype(np.float32))
        }
        
        return torch_data

