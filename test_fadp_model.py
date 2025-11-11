#!/usr/bin/env python3
"""
测试FADP训练模型的脚本

功能：
1. 加载训练好的模型checkpoint
2. 对数据集中的样本进行预测
3. 详细对比预测action和真实action
4. 计算各种评估指标
5. 可视化对比结果

用法:
    python test_fadp_model.py --checkpoint data/outputs/.../checkpoints/latest.ckpt
    python test_fadp_model.py --checkpoint latest.ckpt --num-samples 50 --visualize
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
from collections import defaultdict

# 添加项目路径
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.workspace.train_diffusion_unet_fadp_workspace import TrainDiffusionUnetFADPWorkspace
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from diffusion_policy.policy.diffusion_unet_fadp_policy import DiffusionUnetFADPPolicy
from diffusion_policy.dataset.fadp_dataset import FadpDataset
from diffusion_policy.common.pytorch_util import dict_apply


def print_separator(char='=', length=80):
    """打印分隔线"""
    print(char * length)


def load_checkpoint(checkpoint_path):
    """加载训练好的checkpoint"""
    print(f"加载checkpoint: {checkpoint_path}")
    
    # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含OmegaConf的checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']
    
    # 解析配置
    OmegaConf.resolve(cfg)
    
    # 根据配置中的workspace类型选择正确的workspace
    workspace_target = cfg.get('_target_', '')
    
    if 'fadp' in workspace_target.lower():
        print("检测到FADP workspace，使用 TrainDiffusionUnetFADPWorkspace")
        workspace = TrainDiffusionUnetFADPWorkspace(cfg)
    else:
        print("使用默认 TrainDiffusionUnetImageWorkspace")
        workspace = TrainDiffusionUnetImageWorkspace(cfg)
    
    workspace.load_payload(checkpoint, exclude_keys=None, include_keys=None)
    
    # 获取policy
    policy = workspace.model
    if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
        print("使用EMA模型")
        policy = workspace.ema_model
    else:
        print("使用主模型")
    
    policy.eval()
    
    return policy, cfg, workspace


def load_dataset_from_config(cfg):
    """从配置加载数据集"""
    dataset_cfg = cfg.task.dataset
    
    # 解析shape_meta
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    pose_repr = OmegaConf.to_container(cfg.task.get('pose_repr', {}), resolve=True)
    
    # 创建测试数据集（使用验证集）
    dataset = FadpDataset(
        shape_meta=shape_meta,
        dataset_path=str(cfg.task.dataset_path),
        cache_dir=None,
        pose_repr=pose_repr,
        action_padding=dataset_cfg.get('action_padding', False),
        temporally_independent_normalization=dataset_cfg.get('temporally_independent_normalization', False),
        repeat_frame_prob=0.0,  # 测试时不重复帧
        seed=dataset_cfg.get('seed', 42),
        val_ratio=dataset_cfg.get('val_ratio', 0.05),
        max_duration=dataset_cfg.get('max_duration', None)
    )
    
    # 获取验证集
    val_dataset = dataset.get_validation_dataset()
    
    return dataset, val_dataset


def compute_action_metrics(pred_actions, gt_actions, n_robots=1):
    """
    计算action预测指标
    
    Args:
        pred_actions: (N, T, D) 预测的actions，可能是7维、10维或13维
        gt_actions: (N, T, D) 真实的actions
        n_robots: 机器人数量
    
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 整体指标
    mse = np.mean((pred_actions - gt_actions) ** 2)
    mae = np.mean(np.abs(pred_actions - gt_actions))
    rmse = np.sqrt(mse)
    
    metrics['overall_mse'] = mse
    metrics['overall_mae'] = mae
    metrics['overall_rmse'] = rmse
    
    # 检测action维度
    D = pred_actions.shape[-1]
    has_force = False
    
    if D % 13 == 0:
        # 13维格式: 7维action + 6维force
        action_dim = 13
        n_robots = D // 13
        has_force = True
    elif D % 7 == 0:
        action_dim = 7
        n_robots = D // 7
    elif D % 10 == 0:
        action_dim = 10
        n_robots = D // 10
    else:
        action_dim = D
        n_robots = 1
    
    # 分机器人、分维度统计
    for robot_id in range(n_robots):
        start_idx = robot_id * action_dim
        
        if action_dim == 13:
            # 13维格式: 3 pos + 3 rot + 1 gripper + 6 force
            pos_pred = pred_actions[..., start_idx:start_idx+3]
            pos_gt = gt_actions[..., start_idx:start_idx+3]
            
            rot_pred = pred_actions[..., start_idx+3:start_idx+6]
            rot_gt = gt_actions[..., start_idx+3:start_idx+6]
            
            gripper_pred = pred_actions[..., start_idx+6]
            gripper_gt = gt_actions[..., start_idx+6]
            
            force_pred = pred_actions[..., start_idx+7:start_idx+13]
            force_gt = gt_actions[..., start_idx+7:start_idx+13]
            
        elif action_dim == 7:
            # axis-angle格式
            pos_pred = pred_actions[..., start_idx:start_idx+3]
            pos_gt = gt_actions[..., start_idx:start_idx+3]
            
            rot_pred = pred_actions[..., start_idx+3:start_idx+6]
            rot_gt = gt_actions[..., start_idx+3:start_idx+6]
            
            gripper_pred = pred_actions[..., start_idx+6]
            gripper_gt = gt_actions[..., start_idx+6]
            
        elif action_dim == 10:
            # rot6d格式
            pos_pred = pred_actions[..., start_idx:start_idx+3]
            pos_gt = gt_actions[..., start_idx:start_idx+3]
            
            rot_pred = pred_actions[..., start_idx+3:start_idx+9]
            rot_gt = gt_actions[..., start_idx+3:start_idx+9]
            
            gripper_pred = pred_actions[..., start_idx+9]
            gripper_gt = gt_actions[..., start_idx+9]
        else:
            continue
        
        prefix = f'robot{robot_id}_' if n_robots > 1 else ''
        
        # 位置指标
        metrics[f'{prefix}pos_mse'] = np.mean((pos_pred - pos_gt) ** 2)
        metrics[f'{prefix}pos_mae'] = np.mean(np.abs(pos_pred - pos_gt))
        metrics[f'{prefix}pos_rmse'] = np.sqrt(metrics[f'{prefix}pos_mse'])
        
        # L2范数误差
        pos_l2_error = np.linalg.norm(pos_pred - pos_gt, axis=-1)
        metrics[f'{prefix}pos_l2_mean'] = np.mean(pos_l2_error)
        metrics[f'{prefix}pos_l2_std'] = np.std(pos_l2_error)
        metrics[f'{prefix}pos_l2_max'] = np.max(pos_l2_error)
        
        # 旋转指标
        metrics[f'{prefix}rot_mse'] = np.mean((rot_pred - rot_gt) ** 2)
        metrics[f'{prefix}rot_mae'] = np.mean(np.abs(rot_pred - rot_gt))
        metrics[f'{prefix}rot_rmse'] = np.sqrt(metrics[f'{prefix}rot_mse'])
        
        rot_l2_error = np.linalg.norm(rot_pred - rot_gt, axis=-1)
        metrics[f'{prefix}rot_l2_mean'] = np.mean(rot_l2_error)
        metrics[f'{prefix}rot_l2_std'] = np.std(rot_l2_error)
        metrics[f'{prefix}rot_l2_max'] = np.max(rot_l2_error)
        
        # 夹爪指标
        metrics[f'{prefix}gripper_mse'] = np.mean((gripper_pred - gripper_gt) ** 2)
        metrics[f'{prefix}gripper_mae'] = np.mean(np.abs(gripper_pred - gripper_gt))
        metrics[f'{prefix}gripper_rmse'] = np.sqrt(metrics[f'{prefix}gripper_mse'])
        
        # Force指标（如果有）
        if action_dim == 13:
            metrics[f'{prefix}force_mse'] = np.mean((force_pred - force_gt) ** 2)
            metrics[f'{prefix}force_mae'] = np.mean(np.abs(force_pred - force_gt))
            metrics[f'{prefix}force_rmse'] = np.sqrt(metrics[f'{prefix}force_mse'])
            
            # 分解为平移力和力矩
            force_linear_pred = force_pred[..., :3]  # fx, fy, fz
            force_linear_gt = force_gt[..., :3]
            force_angular_pred = force_pred[..., 3:]  # mx, my, mz
            force_angular_gt = force_gt[..., 3:]
            
            metrics[f'{prefix}force_linear_mse'] = np.mean((force_linear_pred - force_linear_gt) ** 2)
            metrics[f'{prefix}force_linear_mae'] = np.mean(np.abs(force_linear_pred - force_linear_gt))
            metrics[f'{prefix}force_angular_mse'] = np.mean((force_angular_pred - force_angular_gt) ** 2)
            metrics[f'{prefix}force_angular_mae'] = np.mean(np.abs(force_angular_pred - force_angular_gt))
            
            # L2范数误差
            force_l2_error = np.linalg.norm(force_pred - force_gt, axis=-1)
            metrics[f'{prefix}force_l2_mean'] = np.mean(force_l2_error)
            metrics[f'{prefix}force_l2_std'] = np.std(force_l2_error)
            metrics[f'{prefix}force_l2_max'] = np.max(force_l2_error)
    
    return metrics


def test_model(policy, dataset, device, num_samples=None, batch_size=16):
    """
    测试模型并收集预测结果
    
    Args:
        policy: 训练好的policy模型
        dataset: 测试数据集
        device: 设备
        num_samples: 测试样本数量
        batch_size: batch大小
    
    Returns:
        results: 包含预测和真实action的字典
    """
    policy.to(device)
    policy.eval()
    
    # 创建dataloader
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    # 使用subset
    indices = np.arange(num_samples)
    subset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"\n开始测试...")
    print(f"  测试样本数: {num_samples}")
    print(f"  Batch大小: {batch_size}")
    print(f"  设备: {device}")
    
    all_pred_actions = []
    all_gt_actions = []
    all_obs_images = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="测试中")):
            # 移动到设备
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            
            # 预测
            result = policy.predict_action(batch['obs'])
            pred_action = result['action_pred']  # (B, T, D)
            gt_action = batch['action']  # (B, T, D)
            
            # 收集结果
            all_pred_actions.append(pred_action.cpu().numpy())
            all_gt_actions.append(gt_action.cpu().numpy())
            
            # 收集第一帧图像用于可视化
            rgb_key = list(batch['obs'].keys())[0]
            images = batch['obs'][rgb_key][:, 0]  # (B, C, H, W) 取第一帧
            all_obs_images.append(images.cpu().numpy())
    
    # 合并结果
    pred_actions = np.concatenate(all_pred_actions, axis=0)  # (N, T, D)
    gt_actions = np.concatenate(all_gt_actions, axis=0)
    obs_images = np.concatenate(all_obs_images, axis=0)  # (N, C, H, W)
    
    print(f"\n收集完成:")
    print(f"  预测actions形状: {pred_actions.shape}")
    print(f"  真实actions形状: {gt_actions.shape}")
    
    return {
        'pred_actions': pred_actions,
        'gt_actions': gt_actions,
        'obs_images': obs_images
    }


def print_metrics(metrics):
    """打印指标"""
    print_separator('-')
    print("评估指标")
    print_separator('-')
    
    # 整体指标
    print("\n整体指标:")
    print(f"  MSE:  {metrics['overall_mse']:.6f}")
    print(f"  MAE:  {metrics['overall_mae']:.6f}")
    print(f"  RMSE: {metrics['overall_rmse']:.6f}")
    
    # 查找机器人数量
    robot_keys = [k for k in metrics.keys() if 'pos_mse' in k]
    
    for key in robot_keys:
        prefix = key.replace('pos_mse', '')
        robot_name = prefix.rstrip('_') if prefix else 'Robot'
        
        print(f"\n{robot_name} 位置误差:")
        print(f"  MSE:  {metrics[f'{prefix}pos_mse']:.6f}")
        print(f"  MAE:  {metrics[f'{prefix}pos_mae']:.6f}")
        print(f"  RMSE: {metrics[f'{prefix}pos_rmse']:.6f}")
        print(f"  L2 误差 - 均值: {metrics[f'{prefix}pos_l2_mean']:.6f}, "
              f"标准差: {metrics[f'{prefix}pos_l2_std']:.6f}, "
              f"最大: {metrics[f'{prefix}pos_l2_max']:.6f}")
        
        print(f"\n{robot_name} 旋转误差:")
        print(f"  MSE:  {metrics[f'{prefix}rot_mse']:.6f}")
        print(f"  MAE:  {metrics[f'{prefix}rot_mae']:.6f}")
        print(f"  RMSE: {metrics[f'{prefix}rot_rmse']:.6f}")
        print(f"  L2 误差 - 均值: {metrics[f'{prefix}rot_l2_mean']:.6f}, "
              f"标准差: {metrics[f'{prefix}rot_l2_std']:.6f}, "
              f"最大: {metrics[f'{prefix}rot_l2_max']:.6f}")
        
        print(f"\n{robot_name} 夹爪误差:")
        print(f"  MSE:  {metrics[f'{prefix}gripper_mse']:.6f}")
        print(f"  MAE:  {metrics[f'{prefix}gripper_mae']:.6f}")
        print(f"  RMSE: {metrics[f'{prefix}gripper_rmse']:.6f}")
        
        # Force指标（如果有）
        if f'{prefix}force_mse' in metrics:
            print(f"\n{robot_name} Force误差:")
            print(f"  MSE:  {metrics[f'{prefix}force_mse']:.6f}")
            print(f"  MAE:  {metrics[f'{prefix}force_mae']:.6f}")
            print(f"  RMSE: {metrics[f'{prefix}force_rmse']:.6f}")
            print(f"  L2 误差 - 均值: {metrics[f'{prefix}force_l2_mean']:.6f}, "
                  f"标准差: {metrics[f'{prefix}force_l2_std']:.6f}, "
                  f"最大: {metrics[f'{prefix}force_l2_max']:.6f}")
            
            print(f"\n{robot_name} Force分量误差:")
            print(f"  平移力 (fx,fy,fz) - MSE: {metrics[f'{prefix}force_linear_mse']:.6f}, "
                  f"MAE: {metrics[f'{prefix}force_linear_mae']:.6f}")
            print(f"  力矩 (mx,my,mz) - MSE: {metrics[f'{prefix}force_angular_mse']:.6f}, "
                  f"MAE: {metrics[f'{prefix}force_angular_mae']:.6f}")


def visualize_action_comparison(results, save_dir='test_results', num_vis=5):
    """Visualize action comparison"""
    print_separator('-')
    print(f"Generating visualization results (saving to {save_dir})")
    print_separator('-')
    
    os.makedirs(save_dir, exist_ok=True)
    
    pred_actions = results['pred_actions']
    gt_actions = results['gt_actions']
    obs_images = results['obs_images']
    
    N, T, D = pred_actions.shape
    num_vis = min(num_vis, N)
    
    # 检测action维度
    has_force = False
    if D % 13 == 0:
        action_dim = 13
        n_robots = D // 13
        has_force = True
    elif D % 7 == 0:
        action_dim = 7
        n_robots = D // 7
    elif D % 10 == 0:
        action_dim = 10
        n_robots = D // 10
    else:
        action_dim = D
        n_robots = 1
    
    # 1. 绘制action轨迹对比
    for sample_idx in range(num_vis):
        # 根据是否有force调整子图数量
        n_rows = 4 if has_force else 3
        fig, axes = plt.subplots(n_rows, n_robots, figsize=(6*n_robots, 4*n_rows))
        if n_robots == 1:
            axes = axes.reshape(n_rows, 1)
        
        for robot_id in range(n_robots):
            start_idx = robot_id * action_dim
            
            if action_dim == 13:
                # 13维: 3 pos + 3 rot + 1 gripper + 6 force
                pos_pred = pred_actions[sample_idx, :, start_idx:start_idx+3]
                pos_gt = gt_actions[sample_idx, :, start_idx:start_idx+3]
                rot_pred = pred_actions[sample_idx, :, start_idx+3:start_idx+6]
                rot_gt = gt_actions[sample_idx, :, start_idx+3:start_idx+6]
                gripper_pred = pred_actions[sample_idx, :, start_idx+6]
                gripper_gt = gt_actions[sample_idx, :, start_idx+6]
                force_pred = pred_actions[sample_idx, :, start_idx+7:start_idx+13]
                force_gt = gt_actions[sample_idx, :, start_idx+7:start_idx+13]
            elif action_dim == 7:
                pos_pred = pred_actions[sample_idx, :, start_idx:start_idx+3]
                pos_gt = gt_actions[sample_idx, :, start_idx:start_idx+3]
                rot_pred = pred_actions[sample_idx, :, start_idx+3:start_idx+6]
                rot_gt = gt_actions[sample_idx, :, start_idx+3:start_idx+6]
                gripper_pred = pred_actions[sample_idx, :, start_idx+6]
                gripper_gt = gt_actions[sample_idx, :, start_idx+6]
            else:
                # action_dim == 10
                pos_pred = pred_actions[sample_idx, :, start_idx:start_idx+3]
                pos_gt = gt_actions[sample_idx, :, start_idx:start_idx+3]
                rot_pred = pred_actions[sample_idx, :, start_idx+3:start_idx+9]
                rot_gt = gt_actions[sample_idx, :, start_idx+3:start_idx+9]
                gripper_pred = pred_actions[sample_idx, :, start_idx+9]
                gripper_gt = gt_actions[sample_idx, :, start_idx+9]
            
            # Position
            axes[0, robot_id].plot(pos_gt[:, 0], label='GT x', linestyle='--')
            axes[0, robot_id].plot(pos_pred[:, 0], label='Pred x')
            axes[0, robot_id].plot(pos_gt[:, 1], label='GT y', linestyle='--')
            axes[0, robot_id].plot(pos_pred[:, 1], label='Pred y')
            axes[0, robot_id].plot(pos_gt[:, 2], label='GT z', linestyle='--')
            axes[0, robot_id].plot(pos_pred[:, 2], label='Pred z')
            axes[0, robot_id].set_title(f'Robot {robot_id} - Position')
            axes[0, robot_id].set_xlabel('Time Step')
            axes[0, robot_id].set_ylabel('Position Value')
            axes[0, robot_id].legend()
            axes[0, robot_id].grid(True, alpha=0.3)
            
            # Rotation
            for i in range(rot_gt.shape[1]):
                axes[1, robot_id].plot(rot_gt[:, i], label=f'GT rot{i}', linestyle='--', alpha=0.7)
                axes[1, robot_id].plot(rot_pred[:, i], label=f'Pred rot{i}', alpha=0.7)
            axes[1, robot_id].set_title(f'Robot {robot_id} - Rotation')
            axes[1, robot_id].set_xlabel('Time Step')
            axes[1, robot_id].set_ylabel('Rotation Value')
            axes[1, robot_id].legend()
            axes[1, robot_id].grid(True, alpha=0.3)
            
            # Gripper
            axes[2, robot_id].plot(gripper_gt, label='GT', linestyle='--', linewidth=2)
            axes[2, robot_id].plot(gripper_pred, label='Pred', linewidth=2)
            axes[2, robot_id].set_title(f'Robot {robot_id} - Gripper')
            axes[2, robot_id].set_xlabel('Time Step')
            axes[2, robot_id].set_ylabel('Gripper Value')
            axes[2, robot_id].legend()
            axes[2, robot_id].grid(True, alpha=0.3)
            
            # Force (if available)
            if has_force:
                # 绘制6个force维度: fx, fy, fz, mx, my, mz
                force_labels = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
                for i in range(6):
                    axes[3, robot_id].plot(force_gt[:, i], label=f'GT {force_labels[i]}', 
                                          linestyle='--', alpha=0.7)
                    axes[3, robot_id].plot(force_pred[:, i], label=f'Pred {force_labels[i]}', 
                                          alpha=0.7)
                axes[3, robot_id].set_title(f'Robot {robot_id} - Force')
                axes[3, robot_id].set_xlabel('Time Step')
                axes[3, robot_id].set_ylabel('Force Value')
                axes[3, robot_id].legend(ncol=2, fontsize=8)
                axes[3, robot_id].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/action_trajectory_sample_{sample_idx}.png', dpi=150)
        plt.close()
    
    # 2. Plot error distribution
    errors = pred_actions - gt_actions
    
    # 根据是否有force调整子图布局
    if has_force:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Overall error distribution
    axes[0, 0].hist(errors.flatten(), bins=50, edgecolor='black')
    axes[0, 0].set_title('Overall Error Distribution')
    axes[0, 0].set_xlabel('Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error by dimension
    for robot_id in range(n_robots):
        start_idx = robot_id * action_dim
        if action_dim == 13:
            pos_errors = errors[:, :, start_idx:start_idx+3]
            rot_errors = errors[:, :, start_idx+3:start_idx+6]
            gripper_errors = errors[:, :, start_idx+6]
            force_errors = errors[:, :, start_idx+7:start_idx+13]
        elif action_dim == 7:
            pos_errors = errors[:, :, start_idx:start_idx+3]
            rot_errors = errors[:, :, start_idx+3:start_idx+6]
            gripper_errors = errors[:, :, start_idx+6]
        else:
            pos_errors = errors[:, :, start_idx:start_idx+3]
            rot_errors = errors[:, :, start_idx+3:start_idx+9]
            gripper_errors = errors[:, :, start_idx+9]
        
        label_prefix = f'R{robot_id} ' if n_robots > 1 else ''
        
        axes[0, 1].hist(pos_errors.flatten(), bins=30, alpha=0.5, label=f'{label_prefix}Position')
        
        if has_force:
            axes[0, 2].hist(rot_errors.flatten(), bins=30, alpha=0.5, label=f'{label_prefix}Rotation')
            axes[1, 0].hist(gripper_errors.flatten(), bins=30, alpha=0.5, label=f'{label_prefix}Gripper')
            axes[1, 1].hist(force_errors.flatten(), bins=30, alpha=0.5, label=f'{label_prefix}Force')
        else:
            axes[1, 0].hist(rot_errors.flatten(), bins=30, alpha=0.5, label=f'{label_prefix}Rotation')
            axes[1, 1].hist(gripper_errors.flatten(), bins=30, alpha=0.5, label=f'{label_prefix}Gripper')
    
    axes[0, 1].set_title('Position Error Distribution')
    axes[0, 1].set_xlabel('Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if has_force:
        axes[0, 2].set_title('Rotation Error Distribution')
        axes[0, 2].set_xlabel('Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Gripper Error Distribution')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].set_title('Rotation Error Distribution')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Gripper Error Distribution')
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    if has_force:
        axes[1, 1].set_title('Force Error Distribution')
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加force分量误差分布
        axes[1, 2].clear()
        for robot_id in range(n_robots):
            start_idx = robot_id * action_dim
            force_errors = errors[:, :, start_idx+7:start_idx+13]
            force_linear_errors = force_errors[:, :, :3]  # fx, fy, fz
            force_angular_errors = force_errors[:, :, 3:]  # mx, my, mz
            
            label_prefix = f'R{robot_id} ' if n_robots > 1 else ''
            axes[1, 2].hist(force_linear_errors.flatten(), bins=30, alpha=0.5, 
                           label=f'{label_prefix}Linear Force')
            axes[1, 2].hist(force_angular_errors.flatten(), bins=30, alpha=0.5, 
                           label=f'{label_prefix}Angular Force')
        
        axes[1, 2].set_title('Force Components Error Distribution')
        axes[1, 2].set_xlabel('Error')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distribution.png', dpi=150)
    plt.close()
    
    print(f"Visualization results saved to: {save_dir}/")


@click.command()
@click.option('--checkpoint', '-c', required=True, help='模型checkpoint路径')
@click.option('--num-samples', '-n', default=None, type=int, help='测试样本数量')
@click.option('--batch-size', '-b', default=16, type=int, help='Batch大小')
@click.option('--device', '-d', default='cuda:0', help='设备')
@click.option('--visualize', '-v', is_flag=True, help='生成可视化结果')
@click.option('--save-dir', '-s', default='test_results', help='结果保存目录')
@click.option('--use-train-set', is_flag=True, help='使用训练集而不是验证集进行测试')
def main(checkpoint, num_samples, batch_size, device, visualize, save_dir, use_train_set):
    """
    测试FADP训练模型
    
    示例:
        python test_fadp_model.py -c data/outputs/.../latest.ckpt
        python test_fadp_model.py -c latest.ckpt -n 100 -v
        python test_fadp_model.py -c latest.ckpt --use-train-set  # 使用训练集
        python test_fadp_model.py -c latest.ckpt --device cuda:0 --save-dir my_results --use-train-set
    """
    print_separator('=')
    print("FADP模型测试工具")
    print_separator('=')
    
    # 加载checkpoint
    policy, cfg, workspace = load_checkpoint(checkpoint)
    
    # 加载数据集
    train_dataset, val_dataset = load_dataset_from_config(cfg)
    print(f"\n数据集信息:")
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    
    # 选择数据集进行测试
    if use_train_set:
        test_dataset = train_dataset
        dataset_type = 'train'
        print(f"\n使用训练集进行测试")
    else:
        test_dataset = val_dataset
        dataset_type = 'val'
        print(f"\n使用验证集进行测试")
    
    # 更新保存目录，添加数据集类型后缀
    if save_dir == 'test_results':
        save_dir = f'test_results_{dataset_type}'
    
    # 测试模型
    results = test_model(
        policy=policy,
        dataset=test_dataset,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size
    )
    
    # 计算指标
    print_separator()
    print("计算评估指标...")
    n_robots = 1
    D = results['pred_actions'].shape[-1]
    if D % 13 == 0:
        n_robots = D // 13
    elif D % 7 == 0:
        n_robots = D // 7
    elif D % 10 == 0:
        n_robots = D // 10
    
    metrics = compute_action_metrics(
        results['pred_actions'],
        results['gt_actions'],
        n_robots=n_robots
    )
    
    # 打印指标
    print_metrics(metrics)
    
    # 可视化
    if visualize:
        visualize_action_comparison(results, save_dir=save_dir, num_vis=5)
    
    print_separator('=')
    print("测试完成!")
    print_separator('=')


if __name__ == '__main__':
    main()

