#!/usr/bin/env python3
"""
可视化观测Force和未来Force的对比

目的：
1. 查看obs['force']（历史观测，6步）
2. 查看future_force（未来预测目标，16步）
3. 验证两者的时间关系
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.dataset.fadp_dataset import FadpDataset


def visualize_force_comparison(dataset, sample_idx=0):
    """可视化单个样本的force对比"""
    print("=" * 80)
    print(f"样本 {sample_idx} - Force对比")
    print("=" * 80)
    
    # 获取处理后的样本
    sample = dataset[sample_idx]
    
    # 提取数据
    obs_force = sample['obs']['force'].numpy()  # (T_obs, 6) - 历史
    action = sample['action'].numpy()  # (T_action, 13)
    action_only = action[:, :7]  # (T_action, 7)
    force_delta = action[:, 7:]  # (T_action, 6) - 相对force
    
    print(f"\n数据维度:")
    print(f"  obs_force (历史观测): {obs_force.shape}")
    print(f"  force_delta (未来预测，相对): {force_delta.shape}")
    
    # 重建绝对force
    current_force = obs_force[-1]  # 当前force（基准）
    future_force_abs = force_delta + current_force  # 重建绝对force
    
    print(f"\nForce值:")
    print(f"  current_force (基准): {current_force}")
    print(f"  future_force_abs[0]: {future_force_abs[0]}")
    print(f"  future_force_abs[-1]: {future_force_abs[-1]}")
    
    # 绘制对比图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Sample {sample_idx}: Observation Force vs Future Force', fontsize=16, fontweight='bold')
    
    force_labels = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
    
    for i, (ax, label) in enumerate(zip(axes.flat, force_labels)):
        # 观测force（历史）
        T_obs = obs_force.shape[0]
        obs_time = np.arange(-T_obs, 0)  # 负时间表示历史
        ax.plot(obs_time, obs_force[:, i], 'go-', label='Obs Force (History)', 
                linewidth=2, markersize=8, alpha=0.8)
        
        # 当前时刻（分界点）
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                   label='Current Time', alpha=0.7)
        ax.axhline(y=current_force[i], color='red', linestyle=':', 
                   linewidth=1, alpha=0.5)
        
        # 未来force（绝对值）
        T_future = future_force_abs.shape[0]
        future_time = np.arange(0, T_future)  # 正时间表示未来
        ax.plot(future_time, future_force_abs[:, i], 'b^-', 
                label='Future Force (Predicted, Absolute)', 
                linewidth=2, markersize=6, alpha=0.8)
        
        # Force delta（相对值）
        ax2 = ax.twinx()
        ax2.plot(future_time, force_delta[:, i], 'r*--', 
                label='Force Delta (Relative)', 
                linewidth=1.5, markersize=5, alpha=0.6)
        ax2.set_ylabel('Force Delta', color='r', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 设置标签
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step (relative to current)', fontsize=10)
        ax.set_ylabel('Absolute Force Value', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 添加背景色区分历史和未来
        ax.axvspan(-T_obs, 0, alpha=0.1, color='green', label='History')
        ax.axvspan(0, T_future, alpha=0.1, color='blue', label='Future')
    
    plt.tight_layout()
    plt.savefig(f'force_obs_future_comparison_sample_{sample_idx}.png', 
                dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: force_obs_future_comparison_sample_{sample_idx}.png")
    plt.close()


def print_detailed_analysis(dataset, sample_idx=0):
    """打印详细分析"""
    print("\n" + "=" * 80)
    print(f"样本 {sample_idx} - 详细分析")
    print("=" * 80)
    
    sample = dataset[sample_idx]
    obs_force = sample['obs']['force'].numpy()
    force_delta = sample['action'][:, 7:].numpy()
    
    current_force = obs_force[-1]
    future_force_abs = force_delta + current_force
    
    print(f"\n1. 时间维度:")
    print(f"   观测force: {obs_force.shape[0]} 步（历史）")
    print(f"   未来force: {force_delta.shape[0]} 步（预测）")
    
    print(f"\n2. 观测Force统计（历史 {obs_force.shape[0]} 步）:")
    print(f"   均值: {obs_force.mean(axis=0)}")
    print(f"   标准差: {obs_force.std(axis=0)}")
    print(f"   范围: [{obs_force.min(axis=0)}, {obs_force.max(axis=0)}]")
    
    print(f"\n3. 未来Force统计（预测 {force_delta.shape[0]} 步，绝对值）:")
    print(f"   均值: {future_force_abs.mean(axis=0)}")
    print(f"   标准差: {future_force_abs.std(axis=0)}")
    print(f"   范围: [{future_force_abs.min(axis=0)}, {future_force_abs.max(axis=0)}]")
    
    print(f"\n4. Force Delta统计（相对值）:")
    print(f"   均值: {force_delta.mean(axis=0)}")
    print(f"   标准差: {force_delta.std(axis=0)}")
    print(f"   范围: [{force_delta.min(axis=0)}, {force_delta.max(axis=0)}]")
    
    # 检查连续性
    print(f"\n5. 连续性检查:")
    print(f"   obs_force最后一步: {obs_force[-1]}")
    print(f"   future_force第一步: {future_force_abs[0]}")
    print(f"   差异: {np.abs(future_force_abs[0] - obs_force[-1])}")
    
    # 检查变化趋势
    obs_change = obs_force[-1] - obs_force[0]
    future_change = future_force_abs[-1] - future_force_abs[0]
    print(f"\n6. 变化趋势:")
    print(f"   观测期间变化: {obs_change}")
    print(f"   预测期间变化: {future_change}")


def compare_multiple_samples(dataset, num_samples=5):
    """对比多个样本"""
    print("\n" + "=" * 80)
    print(f"对比 {num_samples} 个样本的Force")
    print("=" * 80)
    
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        obs_force = sample['obs']['force'].numpy()
        force_delta = sample['action'][:, 7:].numpy()
        
        obs_std = obs_force.std()
        delta_std = force_delta.std()
        delta_nonzero = (np.abs(force_delta) > 1e-6).mean()
        
        print(f"\n样本 {idx}:")
        print(f"  obs_force std: {obs_std:.6f}")
        print(f"  force_delta std: {delta_std:.6f}")
        print(f"  force_delta非零比例: {delta_nonzero*100:.1f}%")
        
        if delta_std < 1e-6:
            print(f"  ⚠️  警告：force_delta几乎没有变化")
        else:
            print(f"  ✓ force_delta有正常变化")


def main():
    print("=" * 80)
    print("观测Force vs 未来Force 可视化工具")
    print("=" * 80)
    
    # 加载配置
    cfg = OmegaConf.load('diffusion_policy/config/task/fadp_force.yaml')
    OmegaConf.resolve(cfg)
    
    print(f"\n配置信息:")
    print(f"  force_obs_horizon: {cfg.force_obs_horizon}")
    print(f"  action_horizon: {cfg.action_horizon}")
    
    # 创建数据集
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    pose_repr = OmegaConf.to_container(cfg.get('pose_repr', {}), resolve=True)
    
    print(f"\n加载数据集...")
    dataset = FadpDataset(
        shape_meta=shape_meta,
        dataset_path=cfg.dataset_path,
        cache_dir=None,
        pose_repr=pose_repr,
        action_padding=False,
        temporally_independent_normalization=False,
        repeat_frame_prob=0.0,
        seed=42,
        val_ratio=0.05,
        max_duration=None,
        pose_noise_scale=0.0,  # 不使用噪声
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 对比多个样本
    compare_multiple_samples(dataset, num_samples=5)
    
    # 详细分析样本0
    print_detailed_analysis(dataset, sample_idx=0)
    
    # 可视化样本0
    visualize_force_comparison(dataset, sample_idx=687)
    
    # 可视化样本1
    visualize_force_comparison(dataset, sample_idx=234)
    
    print("\n" + "=" * 80)
    print("完成！请查看生成的图表：")
    print("  - force_obs_future_comparison_sample_0.png")
    print("  - force_obs_future_comparison_sample_1.png")
    print("=" * 80)


if __name__ == '__main__':
    main()

