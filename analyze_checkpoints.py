#!/usr/bin/env python3
"""
分析多个checkpoint的性能趋势

功能：
1. 扫描指定文件夹下的所有checkpoint文件
2. 对每个checkpoint计算评估指标
3. 绘制指标随epoch变化的趋势图
4. 保存详细的分析报告

用法:
    python analyze_checkpoints.py --checkpoint-dir data/outputs/.../checkpoints
    python analyze_checkpoints.py -d data/outputs/.../checkpoints --num-samples 100
"""

import sys
import os
import pathlib
import re
import click
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
import json
from collections import defaultdict

# 添加项目路径
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.workspace.train_diffusion_unet_fadp_workspace import TrainDiffusionUnetFADPWorkspace
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy.dataset.fadp_dataset import FadpDataset
from diffusion_policy.common.pytorch_util import dict_apply


def print_separator(char='=', length=80):
    """打印分隔线"""
    print(char * length)


def extract_epoch_from_filename(filename):
    """从文件名中提取epoch数字"""
    # 匹配类似 "epoch_0100.ckpt" 或 "epoch=100-*.ckpt" 的模式
    patterns = [
        r'epoch[_=](\d+)',
        r'step[_=](\d+)',
        r'(\d+)\.ckpt'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    return None


def find_checkpoints(checkpoint_dir):
    """查找目录下所有的checkpoint文件"""
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    
    # 查找所有.ckpt文件
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    
    # 提取epoch信息并排序
    checkpoints = []
    for ckpt_file in ckpt_files:
        epoch = extract_epoch_from_filename(ckpt_file.name)
        if epoch is not None:
            checkpoints.append({
                'path': str(ckpt_file),
                'epoch': epoch,
                'name': ckpt_file.name
            })
        elif 'latest' in ckpt_file.name.lower():
            # latest文件特殊处理，放在最后
            checkpoints.append({
                'path': str(ckpt_file),
                'epoch': float('inf'),
                'name': ckpt_file.name
            })
    
    # 按epoch排序
    checkpoints.sort(key=lambda x: x['epoch'])
    
    return checkpoints


def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']
    OmegaConf.resolve(cfg)
    
    workspace_target = cfg.get('_target_', '')
    
    if 'fadp' in workspace_target.lower():
        workspace = TrainDiffusionUnetFADPWorkspace(cfg)
    else:
        workspace = TrainDiffusionUnetImageWorkspace(cfg)
    
    workspace.load_payload(checkpoint, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
        policy = workspace.ema_model
    
    policy.eval()
    
    return policy, cfg, workspace


def load_dataset_from_config(cfg):
    """从配置加载数据集"""
    dataset_cfg = cfg.task.dataset
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    pose_repr = OmegaConf.to_container(cfg.task.get('pose_repr', {}), resolve=True)
    
    dataset = FadpDataset(
        shape_meta=shape_meta,
        dataset_path=str(cfg.task.dataset_path),
        cache_dir=None,
        pose_repr=pose_repr,
        action_padding=dataset_cfg.get('action_padding', False),
        temporally_independent_normalization=dataset_cfg.get('temporally_independent_normalization', False),
        repeat_frame_prob=0.0,
        seed=dataset_cfg.get('seed', 42),
        val_ratio=dataset_cfg.get('val_ratio', 0.05),
        max_duration=dataset_cfg.get('max_duration', None)
    )
    
    val_dataset = dataset.get_validation_dataset()
    
    return dataset, val_dataset


def compute_action_metrics(pred_actions, gt_actions, n_robots=1):
    """计算action预测指标"""
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
            pos_pred = pred_actions[..., start_idx:start_idx+3]
            pos_gt = gt_actions[..., start_idx:start_idx+3]
            rot_pred = pred_actions[..., start_idx+3:start_idx+6]
            rot_gt = gt_actions[..., start_idx+3:start_idx+6]
            gripper_pred = pred_actions[..., start_idx+6]
            gripper_gt = gt_actions[..., start_idx+6]
            force_pred = pred_actions[..., start_idx+7:start_idx+13]
            force_gt = gt_actions[..., start_idx+7:start_idx+13]
        elif action_dim == 7:
            pos_pred = pred_actions[..., start_idx:start_idx+3]
            pos_gt = gt_actions[..., start_idx:start_idx+3]
            rot_pred = pred_actions[..., start_idx+3:start_idx+6]
            rot_gt = gt_actions[..., start_idx+3:start_idx+6]
            gripper_pred = pred_actions[..., start_idx+6]
            gripper_gt = gt_actions[..., start_idx+6]
        elif action_dim == 10:
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
        
        # Force指标
        if action_dim == 13:
            metrics[f'{prefix}force_mse'] = np.mean((force_pred - force_gt) ** 2)
            metrics[f'{prefix}force_mae'] = np.mean(np.abs(force_pred - force_gt))
            metrics[f'{prefix}force_rmse'] = np.sqrt(metrics[f'{prefix}force_mse'])
            
            force_linear_pred = force_pred[..., :3]
            force_linear_gt = force_gt[..., :3]
            force_angular_pred = force_pred[..., 3:]
            force_angular_gt = force_gt[..., 3:]
            
            metrics[f'{prefix}force_linear_mse'] = np.mean((force_linear_pred - force_linear_gt) ** 2)
            metrics[f'{prefix}force_linear_mae'] = np.mean(np.abs(force_linear_pred - force_linear_gt))
            metrics[f'{prefix}force_angular_mse'] = np.mean((force_angular_pred - force_angular_gt) ** 2)
            metrics[f'{prefix}force_angular_mae'] = np.mean(np.abs(force_angular_pred - force_angular_gt))
            
            force_l2_error = np.linalg.norm(force_pred - force_gt, axis=-1)
            metrics[f'{prefix}force_l2_mean'] = np.mean(force_l2_error)
            metrics[f'{prefix}force_l2_std'] = np.std(force_l2_error)
            metrics[f'{prefix}force_l2_max'] = np.max(force_l2_error)
    
    return metrics


def test_model(policy, dataset, device, num_samples=None, batch_size=16):
    """测试模型并收集预测结果"""
    policy.to(device)
    policy.eval()
    
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    indices = np.arange(num_samples)
    subset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    all_pred_actions = []
    all_gt_actions = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            
            result = policy.predict_action(batch['obs'])
            pred_action = result['action_pred']
            gt_action = batch['action']
            
            all_pred_actions.append(pred_action.cpu().numpy())
            all_gt_actions.append(gt_action.cpu().numpy())
    
    pred_actions = np.concatenate(all_pred_actions, axis=0)
    gt_actions = np.concatenate(all_gt_actions, axis=0)
    
    return pred_actions, gt_actions


def plot_trends(results, save_dir):
    """绘制指标趋势图"""
    print_separator('-')
    print(f"绘制趋势图 (保存到 {save_dir})")
    print_separator('-')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取所有epoch和指标
    epochs = [r['epoch'] for r in results if r['epoch'] != float('inf')]
    
    # 确定有哪些指标
    sample_metrics = results[0]['metrics']
    has_force = 'force_mse' in sample_metrics
    
    # 1. 整体指标趋势
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    overall_mse = [r['metrics']['overall_mse'] for r in results if r['epoch'] != float('inf')]
    overall_mae = [r['metrics']['overall_mae'] for r in results if r['epoch'] != float('inf')]
    overall_rmse = [r['metrics']['overall_rmse'] for r in results if r['epoch'] != float('inf')]
    
    axes[0, 0].plot(epochs, overall_mse, marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('MSE', fontsize=12)
    axes[0, 0].set_title('Overall MSE vs Epoch', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, overall_mae, marker='o', linewidth=2, markersize=4, color='orange')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE', fontsize=12)
    axes[0, 1].set_title('Overall MAE vs Epoch', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, overall_rmse, marker='o', linewidth=2, markersize=4, color='green')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('RMSE', fontsize=12)
    axes[1, 0].set_title('Overall RMSE vs Epoch', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 标记最佳epoch
    best_mse_idx = np.argmin(overall_mse)
    best_mae_idx = np.argmin(overall_mae)
    best_rmse_idx = np.argmin(overall_rmse)
    
    axes[0, 0].scatter([epochs[best_mse_idx]], [overall_mse[best_mse_idx]], 
                       color='red', s=100, zorder=5, label=f'Best (Epoch {epochs[best_mse_idx]})')
    axes[0, 0].legend()
    
    axes[0, 1].scatter([epochs[best_mae_idx]], [overall_mae[best_mae_idx]], 
                       color='red', s=100, zorder=5, label=f'Best (Epoch {epochs[best_mae_idx]})')
    axes[0, 1].legend()
    
    axes[1, 0].scatter([epochs[best_rmse_idx]], [overall_rmse[best_rmse_idx]], 
                       color='red', s=100, zorder=5, label=f'Best (Epoch {epochs[best_rmse_idx]})')
    axes[1, 0].legend()
    
    # 在第四个子图显示统计信息
    axes[1, 1].axis('off')
    stats_text = f"""Best Performance:

MSE:  {min(overall_mse):.6f} @ Epoch {epochs[best_mse_idx]}
MAE:  {min(overall_mae):.6f} @ Epoch {epochs[best_mae_idx]}
RMSE: {min(overall_rmse):.6f} @ Epoch {epochs[best_rmse_idx]}

Total Checkpoints: {len(epochs)}
Epochs Range: {min(epochs)} - {max(epochs)}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/overall_trends.png', dpi=150)
    plt.close()
    
    # 2. 分组件趋势 (Position, Rotation, Gripper)
    n_plots = 4 if has_force else 3
    fig, axes = plt.subplots(2, 2 if n_plots == 4 else 2, figsize=(15, 10))
    axes = axes.flatten()
    
    pos_mse = [r['metrics']['pos_mse'] for r in results if r['epoch'] != float('inf')]
    rot_mse = [r['metrics']['rot_mse'] for r in results if r['epoch'] != float('inf')]
    gripper_mse = [r['metrics']['gripper_mse'] for r in results if r['epoch'] != float('inf')]
    
    axes[0].plot(epochs, pos_mse, marker='o', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Position MSE vs Epoch', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    best_idx = np.argmin(pos_mse)
    axes[0].scatter([epochs[best_idx]], [pos_mse[best_idx]], 
                    color='red', s=100, zorder=5, label=f'Best: {min(pos_mse):.6f}')
    axes[0].legend()
    
    axes[1].plot(epochs, rot_mse, marker='o', linewidth=2, markersize=4, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MSE', fontsize=12)
    axes[1].set_title('Rotation MSE vs Epoch', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    best_idx = np.argmin(rot_mse)
    axes[1].scatter([epochs[best_idx]], [rot_mse[best_idx]], 
                    color='red', s=100, zorder=5, label=f'Best: {min(rot_mse):.6f}')
    axes[1].legend()
    
    axes[2].plot(epochs, gripper_mse, marker='o', linewidth=2, markersize=4, color='green')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('MSE', fontsize=12)
    axes[2].set_title('Gripper MSE vs Epoch', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    best_idx = np.argmin(gripper_mse)
    axes[2].scatter([epochs[best_idx]], [gripper_mse[best_idx]], 
                    color='red', s=100, zorder=5, label=f'Best: {min(gripper_mse):.6f}')
    axes[2].legend()
    
    if has_force:
        force_mse = [r['metrics']['force_mse'] for r in results if r['epoch'] != float('inf')]
        axes[3].plot(epochs, force_mse, marker='o', linewidth=2, markersize=4, color='purple')
        axes[3].set_xlabel('Epoch', fontsize=12)
        axes[3].set_ylabel('MSE', fontsize=12)
        axes[3].set_title('Force MSE vs Epoch', fontsize=14, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        best_idx = np.argmin(force_mse)
        axes[3].scatter([epochs[best_idx]], [force_mse[best_idx]], 
                        color='red', s=100, zorder=5, label=f'Best: {min(force_mse):.6f}')
        axes[3].legend()
    else:
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/component_trends.png', dpi=150)
    plt.close()
    
    # 3. L2误差趋势
    fig, axes = plt.subplots(1, 3 if has_force else 2, figsize=(15 if has_force else 10, 5))
    
    pos_l2 = [r['metrics']['pos_l2_mean'] for r in results if r['epoch'] != float('inf')]
    rot_l2 = [r['metrics']['rot_l2_mean'] for r in results if r['epoch'] != float('inf')]
    
    axes[0].plot(epochs, pos_l2, marker='o', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('L2 Error', fontsize=12)
    axes[0].set_title('Position L2 Error vs Epoch', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    best_idx = np.argmin(pos_l2)
    axes[0].scatter([epochs[best_idx]], [pos_l2[best_idx]], 
                    color='red', s=100, zorder=5, label=f'Best: {min(pos_l2):.6f}')
    axes[0].legend()
    
    axes[1].plot(epochs, rot_l2, marker='o', linewidth=2, markersize=4, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('L2 Error', fontsize=12)
    axes[1].set_title('Rotation L2 Error vs Epoch', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    best_idx = np.argmin(rot_l2)
    axes[1].scatter([epochs[best_idx]], [rot_l2[best_idx]], 
                    color='red', s=100, zorder=5, label=f'Best: {min(rot_l2):.6f}')
    axes[1].legend()
    
    if has_force:
        force_l2 = [r['metrics']['force_l2_mean'] for r in results if r['epoch'] != float('inf')]
        axes[2].plot(epochs, force_l2, marker='o', linewidth=2, markersize=4, color='purple')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('L2 Error', fontsize=12)
        axes[2].set_title('Force L2 Error vs Epoch', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        best_idx = np.argmin(force_l2)
        axes[2].scatter([epochs[best_idx]], [force_l2[best_idx]], 
                        color='red', s=100, zorder=5, label=f'Best: {min(force_l2):.6f}')
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/l2_error_trends.png', dpi=150)
    plt.close()
    
    print(f"趋势图已保存到: {save_dir}/")


@click.command()
@click.option('--checkpoint-dir', '-d', required=True, help='Checkpoint文件夹路径')
@click.option('--num-samples', '-n', default=None, type=int, help='每个checkpoint的测试样本数量')
@click.option('--batch-size', '-b', default=16, type=int, help='Batch大小')
@click.option('--device', default='cuda:0', help='设备')
@click.option('--save-dir', '-s', default='checkpoint_analysis', help='结果保存目录')
@click.option('--use-train-set', is_flag=True, help='使用训练集而不是验证集')
def main(checkpoint_dir, num_samples, batch_size, device, save_dir, use_train_set):
    """
    分析多个checkpoint的性能趋势
    
    示例:
        python analyze_checkpoints.py -d data/outputs/.../checkpoints
        python analyze_checkpoints.py -d data/outputs/.../checkpoints -n 100
        python analyze_checkpoints.py -d data/outputs/.../checkpoints --use-train-set
    """
    print_separator('=')
    print("Checkpoint性能趋势分析工具")
    print_separator('=')
    
    # 查找所有checkpoint
    print(f"\n扫描目录: {checkpoint_dir}")
    checkpoints = find_checkpoints(checkpoint_dir)
    
    if len(checkpoints) == 0:
        print("❌ 未找到任何checkpoint文件！")
        return
    
    print(f"找到 {len(checkpoints)} 个checkpoint文件:")
    for ckpt in checkpoints:
        epoch_str = f"Epoch {ckpt['epoch']}" if ckpt['epoch'] != float('inf') else "Latest"
        print(f"  - {ckpt['name']}: {epoch_str}")
    
    # 加载第一个checkpoint以获取数据集配置
    print(f"\n加载数据集配置...")
    _, cfg, _ = load_checkpoint(checkpoints[0]['path'])
    train_dataset, val_dataset = load_dataset_from_config(cfg)
    
    test_dataset = train_dataset if use_train_set else val_dataset
    dataset_type = 'train' if use_train_set else 'val'
    
    print(f"数据集信息:")
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    print(f"  使用数据集: {dataset_type}")
    
    # 对每个checkpoint进行测试
    results = []
    
    print_separator()
    print("开始测试所有checkpoint...")
    print_separator()
    
    for ckpt_info in tqdm(checkpoints, desc="处理checkpoints"):
        epoch = ckpt_info['epoch']
        ckpt_path = ckpt_info['path']
        
        epoch_str = f"Epoch {epoch}" if epoch != float('inf') else "Latest"
        print(f"\n测试 {ckpt_info['name']} ({epoch_str})")
        
        try:
            # 加载模型
            policy, _, _ = load_checkpoint(ckpt_path)
            
            # 测试模型
            pred_actions, gt_actions = test_model(
                policy=policy,
                dataset=test_dataset,
                device=device,
                num_samples=num_samples,
                batch_size=batch_size
            )
            
            # 计算指标
            D = pred_actions.shape[-1]
            if D % 13 == 0:
                n_robots = D // 13
            elif D % 7 == 0:
                n_robots = D // 7
            elif D % 10 == 0:
                n_robots = D // 10
            else:
                n_robots = 1
            
            metrics = compute_action_metrics(pred_actions, gt_actions, n_robots=n_robots)
            
            results.append({
                'epoch': epoch,
                'name': ckpt_info['name'],
                'path': ckpt_path,
                'metrics': metrics
            })
            
            print(f"  ✓ MSE: {metrics['overall_mse']:.6f}, "
                  f"MAE: {metrics['overall_mae']:.6f}, "
                  f"RMSE: {metrics['overall_rmse']:.6f}")
            
            # 清理GPU内存
            del policy
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            continue
    
    if len(results) == 0:
        print("\n❌ 没有成功处理任何checkpoint！")
        return
    
    # 创建保存目录
    save_dir = f"{save_dir}_{dataset_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制趋势图
    plot_trends(results, save_dir)
    
    # 保存详细结果到JSON
    json_results = []
    for r in results:
        if r['epoch'] != float('inf'):
            json_results.append({
                'epoch': int(r['epoch']),
                'name': r['name'],
                'metrics': {k: float(v) for k, v in r['metrics'].items()}
            })
    
    with open(f'{save_dir}/detailed_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n详细结果已保存到: {save_dir}/detailed_results.json")
    
    # 打印最佳性能总结
    print_separator()
    print("最佳性能总结")
    print_separator()
    
    valid_results = [r for r in results if r['epoch'] != float('inf')]
    
    if len(valid_results) > 0:
        best_mse = min(valid_results, key=lambda x: x['metrics']['overall_mse'])
        best_mae = min(valid_results, key=lambda x: x['metrics']['overall_mae'])
        best_rmse = min(valid_results, key=lambda x: x['metrics']['overall_rmse'])
        
        print(f"\n最佳 MSE:  {best_mse['metrics']['overall_mse']:.6f}")
        print(f"  Epoch: {best_mse['epoch']}")
        print(f"  文件: {best_mse['name']}")
        
        print(f"\n最佳 MAE:  {best_mae['metrics']['overall_mae']:.6f}")
        print(f"  Epoch: {best_mae['epoch']}")
        print(f"  文件: {best_mae['name']}")
        
        print(f"\n最佳 RMSE: {best_rmse['metrics']['overall_rmse']:.6f}")
        print(f"  Epoch: {best_rmse['epoch']}")
        print(f"  文件: {best_rmse['name']}")
        
        # 如果有force信息
        if 'force_mse' in best_mse['metrics']:
            best_force = min(valid_results, key=lambda x: x['metrics']['force_mse'])
            print(f"\n最佳 Force MSE: {best_force['metrics']['force_mse']:.6f}")
            print(f"  Epoch: {best_force['epoch']}")
            print(f"  文件: {best_force['name']}")
    
    print_separator('=')
    print("分析完成！")
    print(f"所有结果保存在: {save_dir}/")
    print_separator('=')


if __name__ == '__main__':
    main()

