#!/usr/bin/env python3
"""
详细检查FADP数据集的脚本

功能：
1. 加载FADP数据集
2. 显示数据集基本信息
3. 检查数据形状和统计信息
4. 可视化样本数据
5. 检查归一化器
6. 分析action分布

用法:
    python inspect_fadp_dataset.py --config-path diffusion_policy/config/task/fadp.yaml
    python inspect_fadp_dataset.py --config-path diffusion_policy/config/task/fadp.yaml --sample-idx 0
    python inspect_fadp_dataset.py --config-path diffusion_policy/config/task/fadp.yaml --visualize
"""

import sys
import os
import pathlib
import click
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

# 添加项目路径
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.dataset.fadp_dataset import FadpDataset


def print_separator(char='=', length=80):
    """打印分隔线"""
    print(char * length)


def load_dataset_from_config(config_path):
    """从配置文件加载数据集"""
    print(f"从配置文件加载: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # 解析所有插值变量（包括嵌套的插值）
    OmegaConf.resolve(cfg)
    
    # 获取数据集配置
    dataset_cfg = cfg.dataset
    
    # 将shape_meta转换为普通字典，确保所有插值都已解析
    shape_meta = OmegaConf.to_container(dataset_cfg.shape_meta, resolve=True)
    
    # 同样处理pose_repr
    pose_repr = OmegaConf.to_container(dataset_cfg.get('pose_repr', {}), resolve=True)
    
    # 创建数据集
    dataset = FadpDataset(
        shape_meta=shape_meta,
        dataset_path=str(dataset_cfg.dataset_path),
        cache_dir=str(dataset_cfg.cache_dir) if dataset_cfg.cache_dir is not None else None,
        pose_repr=pose_repr,
        action_padding=dataset_cfg.get('action_padding', False),
        temporally_independent_normalization=dataset_cfg.get('temporally_independent_normalization', False),
        repeat_frame_prob=dataset_cfg.get('repeat_frame_prob', 0.0),
        seed=dataset_cfg.get('seed', 42),
        val_ratio=dataset_cfg.get('val_ratio', 0.0),
        max_duration=dataset_cfg.get('max_duration', None)
    )
    
    return dataset, cfg


def show_basic_info(dataset):
    """显示数据集基本信息"""
    print_separator()
    print("数据集基本信息")
    print_separator()
    
    print(f"数据集长度: {len(dataset)} 个样本")
    print(f"机器人数量: {dataset.num_robot}")
    print(f"RGB keys: {dataset.rgb_keys}")
    print(f"Low-dim keys (用于计算): {dataset.sampler_lowdim_keys}")
    
    # ReplayBuffer信息
    print(f"\nReplayBuffer信息:")
    print(f"  Episodes数量: {dataset.replay_buffer.n_episodes}")
    print(f"  总步数: {dataset.replay_buffer.n_steps}")
    
    # Episode统计
    episode_ends = dataset.replay_buffer.episode_ends[:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = episode_ends - episode_starts
    
    print(f"\nEpisode长度统计:")
    print(f"  最小: {episode_lengths.min()}")
    print(f"  最大: {episode_lengths.max()}")
    print(f"  平均: {episode_lengths.mean():.1f}")
    print(f"  中位数: {np.median(episode_lengths):.1f}")
    print(f"  标准差: {episode_lengths.std():.1f}")
    
    # 训练/验证集划分
    train_episodes = (~dataset.val_mask).sum()
    val_episodes = dataset.val_mask.sum()
    print(f"\n训练/验证集划分:")
    print(f"  训练episodes: {train_episodes}")
    print(f"  验证episodes: {val_episodes}")
    
    # Shape meta信息
    print(f"\nShape Meta配置:")
    print(f"  Pose表示: obs={dataset.obs_pose_repr}, action={dataset.action_pose_repr}")
    print(f"  Action horizon: {dataset.key_horizon['action']}")
    print(f"  Image horizon: {dataset.key_horizon.get(dataset.rgb_keys[0], 'N/A')}")


def show_raw_data_info(dataset):
    """显示原始数据信息（从replay_buffer）"""
    print_separator('-')
    print("原始数据信息 (Replay Buffer)")
    print_separator('-')
    
    for key in sorted(dataset.replay_buffer.data.keys()):
        data = dataset.replay_buffer.data[key]
        print(f"\n[{key}]")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        
        if data.dtype in [np.float32, np.float64]:
            # 采样部分数据进行统计
            sample_size = min(1000, data.shape[0])
            sample = data[:sample_size]
            print(f"  数值范围: [{np.min(sample):.4f}, {np.max(sample):.4f}]")
            print(f"  均值: {np.mean(sample, axis=0)}")
            print(f"  标准差: {np.std(sample, axis=0)}")
        elif data.dtype == np.uint8:
            print(f"  数值范围: [0, 255] (图像数据)")


def inspect_sample(dataset, sample_idx=0):
    """检查特定样本"""
    print_separator('-')
    print(f"检查样本 #{sample_idx}")
    print_separator('-')
    
    # 获取样本
    sample = dataset[sample_idx]
    
    print(f"\n样本结构:")
    print(f"  Keys: {list(sample.keys())}")
    
    # 观测数据
    print(f"\n观测数据 (obs):")
    for key, value in sample['obs'].items():
        print(f"  [{key}]")
        print(f"    形状: {value.shape}")
        print(f"    数据类型: {value.dtype}")
        print(f"    数值范围: [{value.min():.4f}, {value.max():.4f}]")
        if len(value.shape) == 4:  # 图像 (T, C, H, W)
            print(f"    均值: {value.mean(dim=(0,2,3))}")  # 每个通道的均值
        else:
            print(f"    均值: {value.mean(dim=0)}")
    
    # Action数据
    print(f"\nAction数据:")
    action = sample['action']
    print(f"  形状: {action.shape}")
    print(f"  数据类型: {action.dtype}")
    print(f"  数值范围: [{action.min():.4f}, {action.max():.4f}]")
    
    # 分解action维度
    T = action.shape[0]
    action_np = action.numpy()
    print(f"\n  Action分解 (相对action):")
    for t in [0, T//2, T-1]:
        print(f"    时刻 t={t}:")
        for robot_id in range(dataset.num_robot):
            start_idx = robot_id * 7
            pos = action_np[t, start_idx:start_idx+3]
            rot = action_np[t, start_idx+3:start_idx+6]
            gripper = action_np[t, start_idx+6]
            print(f"      Robot {robot_id}:")
            print(f"        相对位置 (dx,dy,dz): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            print(f"        相对旋转 (drx,dry,drz): [{rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}]")
            print(f"        夹爪: {gripper:.4f}")


def analyze_action_statistics(dataset, num_samples=100):
    """分析action的统计信息"""
    print_separator('-')
    print(f"Action统计分析 (基于 {num_samples} 个样本)")
    print_separator('-')
    
    # 收集action数据
    actions = []
    print("收集样本中...")
    for i in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[i]
        actions.append(sample['action'].numpy())
    
    actions = np.concatenate(actions, axis=0)  # (N*T, D)
    print(f"\n收集到的action数据形状: {actions.shape}")
    
    # 分析每个维度
    dim_names = []
    for robot_id in range(dataset.num_robot):
        prefix = f"Robot{robot_id}_" if dataset.num_robot > 1 else ""
        dim_names.extend([
            f"{prefix}dx", f"{prefix}dy", f"{prefix}dz",
            f"{prefix}drx", f"{prefix}dry", f"{prefix}drz",
            f"{prefix}gripper"
        ])
    
    print(f"\n各维度统计:")
    print(f"{'维度':<15} {'最小值':>10} {'最大值':>10} {'均值':>10} {'标准差':>10}")
    print("-" * 60)
    for i, name in enumerate(dim_names):
        data = actions[:, i]
        print(f"{name:<15} {data.min():>10.4f} {data.max():>10.4f} "
              f"{data.mean():>10.4f} {data.std():>10.4f}")
    
    # 分析位置、旋转、夹爪的整体统计
    print(f"\n分组统计:")
    for robot_id in range(dataset.num_robot):
        start_idx = robot_id * 7
        prefix = f"Robot{robot_id}" if dataset.num_robot > 1 else "Robot"
        
        pos = actions[:, start_idx:start_idx+3]
        rot = actions[:, start_idx+3:start_idx+6]
        gripper = actions[:, start_idx+6]
        
        print(f"\n{prefix}:")
        print(f"  相对位置 (dx,dy,dz):")
        print(f"    L2 norm 范围: [{np.linalg.norm(pos, axis=1).min():.4f}, "
              f"{np.linalg.norm(pos, axis=1).max():.4f}]")
        print(f"    L2 norm 均值: {np.linalg.norm(pos, axis=1).mean():.4f}")
        
        print(f"  相对旋转 (drx,dry,drz):")
        print(f"    L2 norm 范围: [{np.linalg.norm(rot, axis=1).min():.4f}, "
              f"{np.linalg.norm(rot, axis=1).max():.4f}]")
        print(f"    L2 norm 均值: {np.linalg.norm(rot, axis=1).mean():.4f}")
        
        print(f"  夹爪:")
        print(f"    范围: [{gripper.min():.4f}, {gripper.max():.4f}]")
        print(f"    均值: {gripper.mean():.4f}")


def check_normalizer(dataset):
    """检查归一化器"""
    print_separator('-')
    print("归一化器信息")
    print_separator('-')
    
    print("计算归一化器（这可能需要一些时间）...")
    normalizer = dataset.get_normalizer()
    
    print(f"\n归一化器包含的keys: {list(normalizer.keys())}")
    
    # Action归一化器
    print(f"\nAction归一化器:")
    action_normalizer = normalizer['action']
    
    if hasattr(action_normalizer, 'params_dict'):
        params = action_normalizer.params_dict
        print(f"  输入维度: {params['input_stats']['min'].shape}")
        print(f"  输出维度: {params['output_stats']['min'].shape}")
        
        # 显示归一化参数
        dim_names = []
        for robot_id in range(dataset.num_robot):
            prefix = f"R{robot_id}_" if dataset.num_robot > 1 else ""
            dim_names.extend([
                f"{prefix}dx", f"{prefix}dy", f"{prefix}dz",
                f"{prefix}drx", f"{prefix}dry", f"{prefix}drz",
                f"{prefix}grip"
            ])
        
        print(f"\n  输入统计 (原始数据):")
        print(f"  {'维度':<10} {'最小值':>12} {'最大值':>12}")
        print("  " + "-" * 40)
        for i, name in enumerate(dim_names):
            min_val = params['input_stats']['min'][i]
            max_val = params['input_stats']['max'][i]
            print(f"  {name:<10} {min_val:>12.4f} {max_val:>12.4f}")
    
    # RGB归一化器
    print(f"\nRGB归一化器:")
    for key in dataset.rgb_keys:
        print(f"  [{key}]: 使用[0,1]归一化")


def visualize_samples(dataset, num_samples=5, save_path='fadp_samples.png'):
    """可视化样本"""
    print_separator('-')
    print(f"可视化 {num_samples} 个样本")
    print_separator('-')
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要matplotlib来可视化，请安装: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # 获取RGB图像
        rgb_key = dataset.rgb_keys[0]
        images = sample['obs'][rgb_key]  # (T, C, H, W)
        action = sample['action'].numpy()  # (T, D)
        
        T = images.shape[0]
        
        # 显示第一帧、中间帧、最后一帧
        for j, t in enumerate([0, T//2, T-1]):
            img = images[t].numpy().transpose(1, 2, 0)  # (H, W, C)
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"Sample {i}, t={t}")
            axes[i, j].axis('off')
            
            # 添加action信息
            if dataset.num_robot == 1:
                pos_norm = np.linalg.norm(action[t, :3])
                rot_norm = np.linalg.norm(action[t, 3:6])
                gripper = action[t, 6]
                info_text = f"pos_norm={pos_norm:.3f}\nrot_norm={rot_norm:.3f}\ngrip={gripper:.3f}"
                axes[i, j].text(5, 5, info_text, color='yellow', 
                               fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")
    plt.close()


def inspect_batch_data(dataset, batch_size=4, num_batches=3):
    """详细检查batch数据"""
    print_separator('-')
    print(f"检查Batch数据 (batch_size={batch_size}, 检查 {num_batches} 个batches)")
    print_separator('-')
    
    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # 使用0避免多进程问题
    )
    
    print(f"\nDataLoader配置:")
    print(f"  数据集长度: {len(dataset)}")
    print(f"  Batch大小: {batch_size}")
    print(f"  总batch数: {len(dataloader)}")
    print(f"  将检查前 {num_batches} 个batches")
    
    # 检查每个batch
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"\n{'='*80}")
        print(f"Batch #{batch_idx}")
        print(f"{'='*80}")
        
        # 基本信息
        print(f"\nBatch结构:")
        print(f"  Keys: {list(batch.keys())}")
        
        # 观测数据 (obs)
        print(f"\n观测数据 (obs):")
        for key, value in batch['obs'].items():
            print(f"  [{key}]")
            print(f"    形状: {value.shape}")
            print(f"    数据类型: {value.dtype}")
            print(f"    数值范围: [{value.min():.4f}, {value.max():.4f}]")
            
            # 如果是图像，显示每个样本的通道均值
            if len(value.shape) == 5:  # (B, T, C, H, W)
                B, T, C, H, W = value.shape
                print(f"    批次维度: B={B}, T={T}, C={C}, H={H}, W={W}")
                print(f"    每个样本的通道均值:")
                for b in range(B):
                    channel_means = value[b].mean(dim=(1, 2, 3))  # (T, C)
                    print(f"      Sample {b}: {channel_means.mean(dim=0).tolist()}")
            elif len(value.shape) == 4:  # (B, T, C, H, W) 或 (B, T, H, W)
                B, T = value.shape[:2]
                print(f"    批次维度: B={B}, T={T}")
        
        # Action数据
        print(f"\nAction数据:")
        action = batch['action']
        print(f"  形状: {action.shape}")
        print(f"  数据类型: {action.dtype}")
        print(f"  数值范围: [{action.min():.4f}, {action.max():.4f}]")
        
        # 详细分析action
        B, T, D = action.shape
        action_np = action.numpy()
        print(f"  批次维度: B={B}, T={T}, D={D}")
        
        print(f"\n  每个样本的Action详情:")
        for b in range(B):
            print(f"\n    Sample {b}:")
            # 显示第一个、中间、最后一个时刻
            for t in [0, T//2, T-1]:
                print(f"      时刻 t={t}:")
                for robot_id in range(dataset.num_robot):
                    start_idx = robot_id * 7
                    pos = action_np[b, t, start_idx:start_idx+3]
                    rot = action_np[b, t, start_idx+3:start_idx+6]
                    gripper = action_np[b, t, start_idx+6]
                    
                    prefix = f"        Robot {robot_id}:" if dataset.num_robot > 1 else "        "
                    print(f"{prefix}")
                    print(f"          相对位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                    print(f"          位置L2范数: {np.linalg.norm(pos):.4f}")
                    print(f"          相对旋转: [{rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}]")
                    print(f"          旋转L2范数: {np.linalg.norm(rot):.4f}")
                    print(f"          夹爪: {gripper:.4f}")
        
        # 统计信息
        print(f"\n  Batch统计信息:")
        # 位置统计
        for robot_id in range(dataset.num_robot):
            start_idx = robot_id * 7
            pos = action_np[:, :, start_idx:start_idx+3]
            rot = action_np[:, :, start_idx+3:start_idx+6]
            gripper = action_np[:, :, start_idx+6]
            
            prefix = f"  Robot {robot_id}" if dataset.num_robot > 1 else "  "
            print(f"{prefix} 相对位置:")
            print(f"    均值: [{pos.mean(axis=(0,1))[0]:.4f}, {pos.mean(axis=(0,1))[1]:.4f}, {pos.mean(axis=(0,1))[2]:.4f}]")
            print(f"    标准差: [{pos.std(axis=(0,1))[0]:.4f}, {pos.std(axis=(0,1))[1]:.4f}, {pos.std(axis=(0,1))[2]:.4f}]")
            print(f"    L2范数范围: [{np.linalg.norm(pos, axis=2).min():.4f}, {np.linalg.norm(pos, axis=2).max():.4f}]")
            
            print(f"{prefix} 相对旋转:")
            print(f"    均值: [{rot.mean(axis=(0,1))[0]:.4f}, {rot.mean(axis=(0,1))[1]:.4f}, {rot.mean(axis=(0,1))[2]:.4f}]")
            print(f"    标准差: [{rot.std(axis=(0,1))[0]:.4f}, {rot.std(axis=(0,1))[1]:.4f}, {rot.std(axis=(0,1))[2]:.4f}]")
            print(f"    L2范数范围: [{np.linalg.norm(rot, axis=2).min():.4f}, {np.linalg.norm(rot, axis=2).max():.4f}]")
            
            print(f"{prefix} 夹爪:")
            print(f"    范围: [{gripper.min():.4f}, {gripper.max():.4f}]")
            print(f"    均值: {gripper.mean():.4f}")
        
        # 显示数据形状总结
        print(f"\n  Batch数据形状总结:")
        print(f"    obs['camera0_rgb']: {batch['obs'][dataset.rgb_keys[0]].shape}")
        print(f"    action: {batch['action'].shape}")
        print(f"    总元素数: obs={batch['obs'][dataset.rgb_keys[0]].numel()}, action={batch['action'].numel()}")


def plot_action_distribution(dataset, num_samples=100, save_path='fadp_action_dist.png'):
    """绘制action分布"""
    print_separator('-')
    print(f"绘制Action分布 (基于 {num_samples} 个样本)")
    print_separator('-')
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要matplotlib来可视化")
        return
    
    # 收集action数据
    actions = []
    for i in tqdm(range(min(num_samples, len(dataset))), desc="收集样本"):
        sample = dataset[i]
        actions.append(sample['action'].numpy())
    
    actions = np.concatenate(actions, axis=0)  # (N, D)
    
    # 为每个机器人创建子图
    fig, axes = plt.subplots(dataset.num_robot, 3, figsize=(15, 5*dataset.num_robot))
    if dataset.num_robot == 1:
        axes = axes.reshape(1, -1)
    
    for robot_id in range(dataset.num_robot):
        start_idx = robot_id * 7
        
        # 位置分布
        pos = actions[:, start_idx:start_idx+3]
        axes[robot_id, 0].hist(pos[:, 0], bins=50, alpha=0.5, label='dx')
        axes[robot_id, 0].hist(pos[:, 1], bins=50, alpha=0.5, label='dy')
        axes[robot_id, 0].hist(pos[:, 2], bins=50, alpha=0.5, label='dz')
        axes[robot_id, 0].set_xlabel('相对位置')
        axes[robot_id, 0].set_ylabel('频数')
        axes[robot_id, 0].set_title(f'Robot {robot_id} - 相对位置分布')
        axes[robot_id, 0].legend()
        axes[robot_id, 0].grid(True, alpha=0.3)
        
        # 旋转分布
        rot = actions[:, start_idx+3:start_idx+6]
        axes[robot_id, 1].hist(rot[:, 0], bins=50, alpha=0.5, label='drx')
        axes[robot_id, 1].hist(rot[:, 1], bins=50, alpha=0.5, label='dry')
        axes[robot_id, 1].hist(rot[:, 2], bins=50, alpha=0.5, label='drz')
        axes[robot_id, 1].set_xlabel('相对旋转 (axis-angle)')
        axes[robot_id, 1].set_ylabel('频数')
        axes[robot_id, 1].set_title(f'Robot {robot_id} - 相对旋转分布')
        axes[robot_id, 1].legend()
        axes[robot_id, 1].grid(True, alpha=0.3)
        
        # 夹爪分布
        gripper = actions[:, start_idx+6]
        axes[robot_id, 2].hist(gripper, bins=50)
        axes[robot_id, 2].set_xlabel('夹爪值')
        axes[robot_id, 2].set_ylabel('频数')
        axes[robot_id, 2].set_title(f'Robot {robot_id} - 夹爪分布')
        axes[robot_id, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Action分布图已保存到: {save_path}")
    plt.close()


@click.command()
@click.option('--config-path', '-c', required=True,
              help='配置文件路径')
@click.option('--sample-idx', '-s', default=None, type=int,
              help='检查特定样本的索引')
@click.option('--num-samples', '-n', default=100, type=int,
              help='用于统计分析的样本数量')
@click.option('--visualize', '-v', is_flag=True,
              help='可视化样本')
@click.option('--check-normalizer', is_flag=True,
              help='检查归一化器（耗时）')
@click.option('--inspect-batches', '-b', is_flag=True,
              help='检查batch数据')
@click.option('--batch-size', default=4, type=int,
              help='检查batch时的batch大小')
@click.option('--num-batches', default=3, type=int,
              help='检查的batch数量')
def main(config_path, sample_idx, num_samples, visualize, check_normalizer, 
         inspect_batches, batch_size, num_batches):
    """
    详细检查FADP数据集的内容
    
    示例:
        python inspect_fadp_dataset.py -c diffusion_policy/config/task/fadp.yaml
        python inspect_fadp_dataset.py -c diffusion_policy/config/task/fadp.yaml -s 0
        python inspect_fadp_dataset.py -c diffusion_policy/config/task/fadp.yaml -b  # 检查batch数据
        python inspect_fadp_dataset.py -c diffusion_policy/config/task/fadp.yaml -b --batch-size 8 --num-batches 5
        python inspect_fadp_dataset.py -c diffusion_policy/config/task/fadp.yaml -v
        python inspect_fadp_dataset.py -c diffusion_policy/config/task/fadp.yaml --check-normalizer
    """
    print_separator('=')
    print("FADP数据集检查工具")
    print_separator('=')
    
    # 加载数据集
    dataset, cfg = load_dataset_from_config(config_path)
    
    # 显示基本信息
    show_basic_info(dataset)
    
    # 显示原始数据信息
    show_raw_data_info(dataset)
    
    # 检查特定样本
    if sample_idx is not None:
        inspect_sample(dataset, sample_idx)
    else:
        # 默认检查第一个样本
        inspect_sample(dataset, 0)
    
    # 统计分析
    analyze_action_statistics(dataset, num_samples)
    
    # 检查归一化器
    if check_normalizer:
        check_normalizer(dataset)
    
    # 检查batch数据
    if inspect_batches:
        inspect_batch_data(dataset, batch_size=batch_size, num_batches=num_batches)
    
    # 可视化
    if visualize:
        visualize_samples(dataset, num_samples=min(5, len(dataset)))
        plot_action_distribution(dataset, num_samples=num_samples)
    
    print_separator('=')
    print("检查完成!")
    print_separator('=')


if __name__ == '__main__':
    main()

