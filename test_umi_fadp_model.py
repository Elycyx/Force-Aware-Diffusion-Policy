#!/usr/bin/env python3
"""
测试UMI训练模型的脚本（使用FADP数据集）

功能：
1. 加载训练好的UMI模型checkpoint
2. 对FADP数据集中的样本进行预测
3. 详细对比预测action和真实action
4. 计算各种评估指标
5. 可视化对比结果

用法:
    python test_umi_fadp_model.py --checkpoint data/outputs/.../checkpoints/latest.ckpt
    python test_umi_fadp_model.py --checkpoint latest.ckpt --num-samples 50 --visualize
    python test_umi_fadp_model.py --checkpoint latest.ckpt --use-train-set  # 使用训练集
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

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from diffusion_policy.dataset.fadp_umi_dataset import FadpUmiDataset
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
    
    # 使用UMI workspace
    print("使用 TrainDiffusionUnetImageWorkspace")
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
    """从配置加载FADP-UMI数据集"""
    dataset_cfg = cfg.task.dataset
    
    # 解析shape_meta
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    pose_repr = OmegaConf.to_container(cfg.task.get('pose_repr', {}), resolve=True)
    
    # 创建测试数据集（使用FadpUmiDataset）
    dataset = FadpUmiDataset(
        shape_meta=shape_meta,
        dataset_path=str(cfg.task.dataset_path),
        cache_dir=dataset_cfg.get('cache_dir', None),
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
    计算action预测指标（UMI格式：10维 rotation_6d）
    
    Args:
        pred_actions: (N, T, D) 预测的actions，10维格式 [pos(3) + rot_6d(6) + gripper(1)]
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
    
    # UMI格式：10维 [pos(3) + rot_6d(6) + gripper(1)]
    D = pred_actions.shape[-1]
    
    if D % 10 == 0:
        action_dim = 10
        n_robots = D // 10
    else:
        # 如果不是10的倍数，报错
        raise ValueError(f"UMI模型期望10维action（或其倍数），但得到{D}维")
    
    # 分机器人、分维度统计
    for robot_id in range(n_robots):
        start_idx = robot_id * action_dim
        
        # UMI格式: 3 pos + 6 rot_6d + 1 gripper
        pos_pred = pred_actions[..., start_idx:start_idx+3]
        pos_gt = gt_actions[..., start_idx:start_idx+3]
        
        rot_pred = pred_actions[..., start_idx+3:start_idx+9]
        rot_gt = gt_actions[..., start_idx+3:start_idx+9]
        
        gripper_pred = pred_actions[..., start_idx+9]
        gripper_gt = gt_actions[..., start_idx+9]
        
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
        
        # 旋转指标（rotation_6d）
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
            # pred_action: predict_action()内部已经反归一化，返回原始值
            pred_action = result['action']  # (B, T, D) - 原始值
            # gt_action: 从dataset来的，也是原始值
            gt_action = batch['action']  # (B, T, D) - 原始值
            
            # 两者都已经是原始值，可以直接对比！
            # 不需要额外的归一化/反归一化操作
            
            # 收集结果
            all_pred_actions.append(pred_action.cpu().numpy())
            all_gt_actions.append(gt_action.cpu().numpy())
            
            # 保存一些观测图像
            if 'camera0_rgb' in batch['obs'] and len(all_obs_images) < 10:
                obs_images = batch['obs']['camera0_rgb'][:, 0].cpu().numpy()  # 取第一帧
                all_obs_images.append(obs_images)
    
    # 合并所有结果
    pred_actions = np.concatenate(all_pred_actions, axis=0)  # (N, T, D)
    gt_actions = np.concatenate(all_gt_actions, axis=0)  # (N, T, D)
    
    if len(all_obs_images) > 0:
        obs_images = np.concatenate(all_obs_images, axis=0)  # (N, C, H, W)
    else:
        obs_images = None
    
    print(f"\n测试完成!")
    print(f"  预测action shape: {pred_actions.shape}")
    print(f"  真实action shape: {gt_actions.shape}")
    
    return {
        'pred_actions': pred_actions,
        'gt_actions': gt_actions,
        'obs_images': obs_images
    }


def print_metrics(metrics):
    """打印指标（格式化输出）"""
    print_separator()
    print("评估指标:")
    print_separator()
    
    # 整体指标
    print("\n【整体指标】")
    print(f"  MSE:  {metrics['overall_mse']:.6f}")
    print(f"  MAE:  {metrics['overall_mae']:.6f}")
    print(f"  RMSE: {metrics['overall_rmse']:.6f}")
    
    # 分维度指标
    robot_keys = [k for k in metrics.keys() if 'robot' in k or k.startswith('pos_') or k.startswith('rot_') or k.startswith('gripper_')]
    if not robot_keys:
        robot_keys = [k for k in metrics.keys() if k not in ['overall_mse', 'overall_mae', 'overall_rmse']]
    
    # 按机器人分组
    robot_ids = set()
    for key in robot_keys:
        if 'robot' in key:
            robot_id = key.split('_')[0]
            robot_ids.add(robot_id)
    
    if not robot_ids:
        robot_ids = ['']
    
    for robot_id in sorted(robot_ids):
        if robot_id:
            print(f"\n【{robot_id.upper()}】")
            prefix = robot_id + '_'
        else:
            print(f"\n【分维度指标】")
            prefix = ''
        
        # Position
        if f'{prefix}pos_mse' in metrics:
            print(f"  Position:")
            print(f"    MSE:      {metrics[f'{prefix}pos_mse']:.6f}")
            print(f"    MAE:      {metrics[f'{prefix}pos_mae']:.6f}")
            print(f"    RMSE:     {metrics[f'{prefix}pos_rmse']:.6f}")
            print(f"    L2 Mean:  {metrics[f'{prefix}pos_l2_mean']:.6f}")
            print(f"    L2 Std:   {metrics[f'{prefix}pos_l2_std']:.6f}")
            print(f"    L2 Max:   {metrics[f'{prefix}pos_l2_max']:.6f}")
        
        # Rotation
        if f'{prefix}rot_mse' in metrics:
            print(f"  Rotation (rotation_6d):")
            print(f"    MSE:      {metrics[f'{prefix}rot_mse']:.6f}")
            print(f"    MAE:      {metrics[f'{prefix}rot_mae']:.6f}")
            print(f"    RMSE:     {metrics[f'{prefix}rot_rmse']:.6f}")
            print(f"    L2 Mean:  {metrics[f'{prefix}rot_l2_mean']:.6f}")
            print(f"    L2 Std:   {metrics[f'{prefix}rot_l2_std']:.6f}")
            print(f"    L2 Max:   {metrics[f'{prefix}rot_l2_max']:.6f}")
        
        # Gripper
        if f'{prefix}gripper_mse' in metrics:
            print(f"  Gripper:")
            print(f"    MSE:      {metrics[f'{prefix}gripper_mse']:.6f}")
            print(f"    MAE:      {metrics[f'{prefix}gripper_mae']:.6f}")
            print(f"    RMSE:     {metrics[f'{prefix}gripper_rmse']:.6f}")


def visualize_action_comparison(results, save_dir='test_results', num_vis=5):
    """
    可视化预测和真实action的对比
    
    Args:
        results: 包含pred_actions和gt_actions的字典
        save_dir: 保存目录
        num_vis: 可视化样本数量
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pred_actions = results['pred_actions']
    gt_actions = results['gt_actions']
    obs_images = results.get('obs_images', None)
    
    N, T, D = pred_actions.shape
    
    # 检测action维度和机器人数量
    if D % 10 == 0:
        action_dim = 10
        n_robots = D // 10
    else:
        raise ValueError(f"UMI模型期望10维action，但得到{D}维")
    
    print(f"\n生成可视化结果...")
    print(f"  保存目录: {save_dir}/")
    print(f"  可视化样本数: {min(num_vis, N)}")
    print(f"  Action维度: {action_dim} (UMI格式: pos(3) + rot_6d(6) + gripper(1))")
    print(f"  机器人数量: {n_robots}")
    
    # 1. Plot action trajectories for selected samples
    num_vis = min(num_vis, N)
    sample_indices = np.linspace(0, N-1, num_vis, dtype=int)
    
    for vis_idx, sample_idx in enumerate(sample_indices):
        # 创建子图：每个机器人3行（position, rotation_6d, gripper）
        fig, axes = plt.subplots(3, n_robots, figsize=(6*n_robots, 12))
        if n_robots == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Sample {sample_idx}: Action Trajectory Comparison', fontsize=16)
        
        for robot_id in range(n_robots):
            start_idx = robot_id * action_dim
            
            # UMI格式: 3 pos + 6 rot_6d + 1 gripper
            pos_gt = gt_actions[sample_idx, :, start_idx:start_idx+3]
            pos_pred = pred_actions[sample_idx, :, start_idx:start_idx+3]
            
            rot_gt = gt_actions[sample_idx, :, start_idx+3:start_idx+9]
            rot_pred = pred_actions[sample_idx, :, start_idx+3:start_idx+9]
            
            gripper_gt = gt_actions[sample_idx, :, start_idx+9]
            gripper_pred = pred_actions[sample_idx, :, start_idx+9]
            
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
            
            # Rotation (rotation_6d: 6 dimensions)
            for i in range(6):
                axes[1, robot_id].plot(rot_gt[:, i], label=f'GT rot{i}', linestyle='--', alpha=0.7)
                axes[1, robot_id].plot(rot_pred[:, i], label=f'Pred rot{i}', alpha=0.7)
            axes[1, robot_id].set_title(f'Robot {robot_id} - Rotation (rotation_6d)')
            axes[1, robot_id].set_xlabel('Time Step')
            axes[1, robot_id].set_ylabel('Rotation_6d Value')
            axes[1, robot_id].legend(ncol=2, fontsize=8)
            axes[1, robot_id].grid(True, alpha=0.3)
            
            # Gripper
            axes[2, robot_id].plot(gripper_gt, label='GT', linestyle='--', linewidth=2)
            axes[2, robot_id].plot(gripper_pred, label='Pred', linewidth=2)
            axes[2, robot_id].set_title(f'Robot {robot_id} - Gripper')
            axes[2, robot_id].set_xlabel('Time Step')
            axes[2, robot_id].set_ylabel('Gripper Value')
            axes[2, robot_id].legend()
            axes[2, robot_id].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/action_trajectory_sample_{vis_idx}_idx{sample_idx}.png', dpi=150)
        plt.close()
        print(f"  已保存样本 {sample_idx} 的可视化结果")
    
    # 2. Plot error distribution
    errors = pred_actions - gt_actions
    
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
        pos_errors = errors[:, :, start_idx:start_idx+3]
        rot_errors = errors[:, :, start_idx+3:start_idx+9]
        gripper_errors = errors[:, :, start_idx+9]
        
        label_prefix = f'R{robot_id} ' if n_robots > 1 else ''
        
        axes[0, 1].hist(pos_errors.flatten(), bins=30, alpha=0.5, label=f'{label_prefix}Position')
        axes[1, 0].hist(rot_errors.flatten(), bins=30, alpha=0.5, label=f'{label_prefix}Rotation_6d')
        axes[1, 1].hist(gripper_errors.flatten(), bins=30, alpha=0.5, label=f'{label_prefix}Gripper')
    
    axes[0, 1].set_title('Position Error Distribution')
    axes[0, 1].set_xlabel('Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Rotation_6d Error Distribution')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Gripper Error Distribution')
    axes[1, 1].set_xlabel('Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
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
@click.option('--save-dir', '-s', default='test_results_umi', help='结果保存目录')
@click.option('--use-train-set', is_flag=True, help='使用训练集而不是验证集进行测试')
def main(checkpoint, num_samples, batch_size, device, visualize, save_dir, use_train_set):
    """
    测试UMI训练模型（使用FADP数据集）
    
    示例:
        python test_umi_fadp_model.py -c data/outputs/.../latest.ckpt
        python test_umi_fadp_model.py -c latest.ckpt -n 100 -v
        python test_umi_fadp_model.py -c latest.ckpt --use-train-set  # 使用训练集
        python test_umi_fadp_model.py -c latest.ckpt --device cuda:0 --save-dir my_results
    """
    print_separator('=')
    print("UMI模型测试工具（FADP数据集）")
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
    if save_dir == 'test_results_umi':
        save_dir = f'test_results_umi_{dataset_type}'
    
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
    D = results['pred_actions'].shape[-1]
    n_robots = D // 10 if D % 10 == 0 else 1
    
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
    print(f"数据集类型: {dataset_type}")
    print(f"结果保存在: {save_dir}/")
    print_separator('=')


if __name__ == '__main__':
    main()
