#!/usr/bin/env python3
"""
开环评估：使用训练后的模型对完整episode进行开环测试并生成视频

开环测试流程：
1. 从数据集获取完整episode的观测序列
2. 逐步使用模型预测action
3. 不反馈预测结果，继续使用数据集的下一帧观测
4. 可视化ground truth vs predicted轨迹
5. 保存为视频文件
"""

import sys
import pathlib
import torch
import numpy as np
import click
import cv2
import zarr
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.workspace.train_diffusion_unet_fadp_workspace import TrainDiffusionUnetFADPWorkspace
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy.policy.diffusion_unet_fadp_policy import DiffusionUnetFADPPolicy


def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']
    
    # 根据配置选择workspace
    if 'fadp' in cfg._target_.lower() or 'fadp' in str(cfg.get('policy', {}).get('_target_', '')).lower():
        workspace_cls = TrainDiffusionUnetFADPWorkspace
    else:
        workspace_cls = TrainDiffusionUnetImageWorkspace
    
    workspace = workspace_cls(cfg)
    workspace.load_payload(checkpoint, exclude_keys=None, include_keys=None)
    
    # 获取policy
    policy = workspace.model
    if workspace.cfg.training.use_ema:
        policy = workspace.ema_model
    policy.eval()
    policy.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return policy, cfg, workspace


class ObservationBuffer:
    """观测缓冲区，维护历史观测"""
    def __init__(self, img_horizon=2, force_horizon=6):
        self.img_horizon = img_horizon
        self.force_horizon = force_horizon
        self.img_buffer = deque(maxlen=img_horizon)
        self.force_buffer = deque(maxlen=force_horizon)
    
    def reset(self):
        """重置缓冲区"""
        self.img_buffer.clear()
        self.force_buffer.clear()
    
    def add(self, img, force=None):
        """添加新观测"""
        self.img_buffer.append(img)
        if force is not None:
            self.force_buffer.append(force)
    
    def get_obs(self):
        """获取当前观测（用于模型输入）"""
        # 填充不足的历史
        while len(self.img_buffer) < self.img_horizon:
            if len(self.img_buffer) > 0:
                self.img_buffer.appendleft(self.img_buffer[0])
            else:
                return None
        
        obs = {}
        
        # 图像观测 (T, C, H, W) -> (1, T, C, H, W)
        img_stack = torch.stack(list(self.img_buffer), dim=0).unsqueeze(0)
        obs['camera0_rgb'] = img_stack
        
        # Force观测（如果有）
        if len(self.force_buffer) > 0:
            while len(self.force_buffer) < self.force_horizon:
                if len(self.force_buffer) > 0:
                    self.force_buffer.appendleft(self.force_buffer[0])
            force_stack = torch.stack(list(self.force_buffer), dim=0).unsqueeze(0)
            obs['force'] = force_stack
        
        return obs


def load_episode_from_zarr(dataset_path, episode_idx):
    """从zarr文件加载完整episode"""
    print(f"\n加载episode {episode_idx} 从 {dataset_path}")
    
    # 打开zarr
    if dataset_path.endswith('.zip'):
        import zipfile
        store = zarr.ZipStore(dataset_path, mode='r')
    else:
        store = zarr.DirectoryStore(dataset_path)
    
    root = zarr.open(store, mode='r')
    
    # 获取episode范围
    episode_ends = root['meta']['episode_ends'][:]
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
    end_idx = episode_ends[episode_idx]
    
    print(f"Episode {episode_idx}: 帧 {start_idx} 到 {end_idx} (共 {end_idx - start_idx} 帧)")
    
    # 加载数据
    data = {
        'camera0_rgb': root['data']['camera0_rgb'][start_idx:end_idx],
        'action': root['data']['action'][start_idx:end_idx],
    }
    
    # 检查是否有force数据
    if 'force' in root['data']:
        data['force'] = root['data']['force'][start_idx:end_idx]
        print(f"包含force数据: {data['force'].shape}")
    
    print(f"RGB shape: {data['camera0_rgb'].shape}")
    print(f"Action shape: {data['action'].shape}")
    
    return data


def run_open_loop_prediction(policy, episode_data, cfg, device='cuda', action_chunk_size=8):
    """
    运行开环预测（使用action chunking）
    
    Args:
        policy: 训练好的策略
        episode_data: episode数据
        cfg: 配置
        device: 设备
        action_chunk_size: action chunking大小（推理一次执行多少步）
    """
    print("\n开始开环预测...")
    
    # 确定配置
    img_horizon = cfg.task.get('img_obs_horizon', 2)
    force_horizon = cfg.task.get('force_obs_horizon', 6)
    action_horizon = cfg.task.get('action_horizon', 16)
    has_force = 'force' in episode_data
    
    # 检查action维度
    action_dim = episode_data['action'].shape[-1]
    is_force_prediction = (action_dim == 13)
    
    print(f"配置: img_horizon={img_horizon}, force_horizon={force_horizon}, action_horizon={action_horizon}")
    print(f"Action维度: {action_dim} ({'13D force prediction' if is_force_prediction else '7D basic'})")
    print(f"Action chunking: 推理一次执行 {action_chunk_size} 步")
    
    # 初始化缓冲区
    obs_buffer = ObservationBuffer(img_horizon=img_horizon, force_horizon=force_horizon)
    
    # 存储结果
    num_frames = len(episode_data['camera0_rgb'])
    predicted_actions = []  # 相对action (delta)
    predicted_forces = [] if is_force_prediction else None
    
    # Action chunking变量
    action_buffer = None
    action_buffer_idx = 0
    
    # 逐帧预测（使用action chunking）
    for t in tqdm(range(num_frames), desc="预测中"):
        # 预处理图像
        img_raw = episode_data['camera0_rgb'][t]  # (H, W, 3)
        img_tensor = torch.from_numpy(img_raw).float().permute(2, 0, 1) / 255.0  # (C, H, W)
        
        # Force数据（如果有）
        force_tensor = None
        if has_force:
            force_raw = episode_data['force'][t]  # (6,)
            force_tensor = torch.from_numpy(force_raw).float()
        
        # 添加到缓冲区
        obs_buffer.add(img_tensor, force_tensor)
        
        # 检查是否需要重新推理
        need_inference = (action_buffer is None or action_buffer_idx >= len(action_buffer))
        
        if need_inference:
            # 获取观测
            obs = obs_buffer.get_obs()
            if obs is None:
                # 历史不足，使用零action
                predicted_actions.append(np.zeros(7))
                if predicted_forces is not None:
                    predicted_forces.append(np.zeros(6))
                continue
            
            # 移动到设备
            obs = {k: v.to(device) for k, v in obs.items()}
            
            # 推理
            with torch.no_grad():
                result = policy.predict_action(obs)
            
            # 提取预测序列
            if is_force_prediction:
                action_seq = result['action_only'][0].cpu().numpy()  # (T_pred, 7)
                force_seq = result['force_pred'][0].cpu().numpy()    # (T_pred, 6)
                # 只缓存前action_chunk_size步
                action_buffer = action_seq[:action_chunk_size]
                force_buffer = force_seq[:action_chunk_size]
            else:
                action_seq = result['action'][0].cpu().numpy()  # (T_pred, 7)
                action_buffer = action_seq[:action_chunk_size]
                force_buffer = None
            
            action_buffer_idx = 0
        
        # 使用缓存的action
        predicted_actions.append(action_buffer[action_buffer_idx])
        if predicted_forces is not None and force_buffer is not None:
            predicted_forces.append(force_buffer[action_buffer_idx])
        
        action_buffer_idx += 1
    
    # 转换为numpy数组
    predicted_actions = np.array(predicted_actions)  # (T, 7) - relative to inference frame
    ground_truth_actions = episode_data['action'][:, :7]  # (T, 7) - relative to previous frame
    
    # 转换为绝对轨迹
    # 关键：模型的action chunk是相对于推理时刻的frame，而不是相对于前一帧
    print("\n转换为绝对轨迹...")
    pred_abs = np.zeros_like(predicted_actions)
    gt_abs = np.zeros_like(ground_truth_actions)
    
    # 初始位置（假设起点为0）
    pred_abs[0] = predicted_actions[0]
    gt_abs[0] = ground_truth_actions[0]
    
    # 对于GT：每帧相对于前一帧，需要累加
    for t in range(1, len(ground_truth_actions)):
        gt_abs[t, :6] = gt_abs[t-1, :6] + ground_truth_actions[t, :6]
        gt_abs[t, 6] = ground_truth_actions[t, 6]
    
    # 对于Pred：需要根据推理时刻来计算
    # 每次推理（action_buffer_idx==0）时，记录当前绝对位置作为参考
    # chunk内的每一步都是相对于这个参考位置的增量
    current_reference_pose = np.zeros(7)  # 当前chunk的参考pose（绝对位置）
    chunk_start_idx = 0  # 当前chunk的起始索引
    
    for t in range(len(predicted_actions)):
        # 检查是否是新的推理周期（根据action_chunk_size判断）
        if t > 0 and t % action_chunk_size == 0:
            # 更新参考pose为当前的绝对位置
            current_reference_pose = pred_abs[t - 1].copy()
            chunk_start_idx = t
        
        # 应用相对于参考pose的action
        pred_abs[t, :6] = current_reference_pose[:6] + predicted_actions[t, :6]
        pred_abs[t, 6] = predicted_actions[t, 6]  # gripper是绝对值
    
    results = {
        'predicted_actions': pred_abs,      # 绝对轨迹
        'ground_truth_actions': gt_abs,     # 绝对轨迹
        'predicted_actions_delta': predicted_actions,  # 相对action（相对于推理时刻）
        'ground_truth_actions_delta': ground_truth_actions,  # 相对action（相对于前一帧）
    }
    
    if is_force_prediction and predicted_forces is not None:
        predicted_forces = np.array(predicted_forces)  # (T, 6)
        ground_truth_forces = episode_data['action'][:, 7:13]  # (T, 6) - relative force
        
        # Force已经是绝对值（从predict_action输出）
        results['predicted_forces'] = predicted_forces
        results['ground_truth_forces_delta'] = ground_truth_forces
        
        # 如果有实际force观测，可以用于对比
        if has_force:
            results['observed_forces'] = episode_data['force']
    
    print(f"预测完成！共 {len(predicted_actions)} 帧")
    print(f"推理次数: 约 {int(np.ceil(num_frames / action_chunk_size))} 次")
    return results


def create_visualization_frame(rgb_img, pred_actions, gt_actions, frame_idx, 
                              pred_forces=None, gt_forces=None,
                              action_errors=None, cum_errors=None, force_errors=None):
    """创建单帧可视化（使用matplotlib）"""
    has_force = (pred_forces is not None and gt_forces is not None)
    
    # 创建figure
    if has_force:
        fig = plt.figure(figsize=(24, 15))
        gs = fig.add_gridspec(5, 4, hspace=0.35, wspace=0.35)
    else:
        fig = plt.figure(figsize=(20, 9))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    
    action_labels = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
    
    # 1. RGB图像 (左上，2x2)
    ax_rgb = fig.add_subplot(gs[:2, 0])
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title(f'Frame {frame_idx}', fontsize=14, fontweight='bold')
    ax_rgb.axis('off')
    
    # 2. Position X (右上)
    ax_x = fig.add_subplot(gs[0, 1])
    ax_x.plot(gt_actions[:, 0], 'b-', label='GT', linewidth=2, alpha=0.8)
    ax_x.plot(pred_actions[:, 0], 'r--', label='Pred', linewidth=2, alpha=0.8)
    ax_x.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax_x.set_title('Position X (m)', fontsize=10, fontweight='bold')
    ax_x.legend(loc='best', fontsize=8)
    ax_x.grid(True, alpha=0.3)
    ax_x.set_xlabel('Frame', fontsize=8)
    
    # 3. Position Y
    ax_y = fig.add_subplot(gs[0, 2])
    ax_y.plot(gt_actions[:, 1], 'b-', label='GT', linewidth=2, alpha=0.8)
    ax_y.plot(pred_actions[:, 1], 'r--', label='Pred', linewidth=2, alpha=0.8)
    ax_y.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax_y.set_title('Position Y (m)', fontsize=10, fontweight='bold')
    ax_y.legend(loc='best', fontsize=8)
    ax_y.grid(True, alpha=0.3)
    ax_y.set_xlabel('Frame', fontsize=8)
    
    # 4. Position Z
    ax_z = fig.add_subplot(gs[0, 3])
    ax_z.plot(gt_actions[:, 2], 'b-', label='GT', linewidth=2, alpha=0.8)
    ax_z.plot(pred_actions[:, 2], 'r--', label='Pred', linewidth=2, alpha=0.8)
    ax_z.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax_z.set_title('Position Z (m)', fontsize=10, fontweight='bold')
    ax_z.legend(loc='best', fontsize=8)
    ax_z.grid(True, alpha=0.3)
    ax_z.set_xlabel('Frame', fontsize=8)
    
    # 5. Rotation RX
    ax_rx = fig.add_subplot(gs[1, 1])
    ax_rx.plot(gt_actions[:, 3], 'b-', label='GT', linewidth=2, alpha=0.8)
    ax_rx.plot(pred_actions[:, 3], 'r--', label='Pred', linewidth=2, alpha=0.8)
    ax_rx.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax_rx.set_title('Rotation RX (rad)', fontsize=10, fontweight='bold')
    ax_rx.legend(loc='best', fontsize=8)
    ax_rx.grid(True, alpha=0.3)
    ax_rx.set_xlabel('Frame', fontsize=8)
    
    # 6. Rotation RY
    ax_ry = fig.add_subplot(gs[1, 2])
    ax_ry.plot(gt_actions[:, 4], 'b-', label='GT', linewidth=2, alpha=0.8)
    ax_ry.plot(pred_actions[:, 4], 'r--', label='Pred', linewidth=2, alpha=0.8)
    ax_ry.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax_ry.set_title('Rotation RY (rad)', fontsize=10, fontweight='bold')
    ax_ry.legend(loc='best', fontsize=8)
    ax_ry.grid(True, alpha=0.3)
    ax_ry.set_xlabel('Frame', fontsize=8)
    
    # 7. Rotation RZ
    ax_rz = fig.add_subplot(gs[1, 3])
    ax_rz.plot(gt_actions[:, 5], 'b-', label='GT', linewidth=2, alpha=0.8)
    ax_rz.plot(pred_actions[:, 5], 'r--', label='Pred', linewidth=2, alpha=0.8)
    ax_rz.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax_rz.set_title('Rotation RZ (rad)', fontsize=10, fontweight='bold')
    ax_rz.legend(loc='best', fontsize=8)
    ax_rz.grid(True, alpha=0.3)
    ax_rz.set_xlabel('Frame', fontsize=8)
    
    # 8. Gripper
    ax_grip = fig.add_subplot(gs[2, 0])
    ax_grip.plot(gt_actions[:, 6], 'b-', label='GT', linewidth=2, alpha=0.8)
    ax_grip.plot(pred_actions[:, 6], 'r--', label='Pred', linewidth=2, alpha=0.8)
    ax_grip.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax_grip.set_title('Gripper', fontsize=10, fontweight='bold')
    ax_grip.legend(loc='best', fontsize=8)
    ax_grip.grid(True, alpha=0.3)
    ax_grip.set_xlabel('Frame', fontsize=8)
    
    # 9. Per-dimension errors
    if action_errors is not None:
        ax_err = fig.add_subplot(gs[2, 1])
        for i, label in enumerate(action_labels):
            ax_err.plot(action_errors[:, i], label=label, linewidth=1.5, alpha=0.7)
        ax_err.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax_err.set_title('Per-Dimension Absolute Error', fontsize=10, fontweight='bold')
        ax_err.legend(loc='best', fontsize=7, ncol=2)
        ax_err.grid(True, alpha=0.3)
        ax_err.set_yscale('log')
        ax_err.set_xlabel('Frame', fontsize=8)
        ax_err.set_ylabel('Error (log scale)', fontsize=8)
    
    # 10. Cumulative errors
    if cum_errors is not None:
        ax_cum = fig.add_subplot(gs[2, 2])
        for i, label in enumerate(action_labels):
            ax_cum.plot(cum_errors[:, i], label=label, linewidth=1.5, alpha=0.7)
        ax_cum.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax_cum.set_title('Cumulative Error', fontsize=10, fontweight='bold')
        ax_cum.legend(loc='best', fontsize=7, ncol=2)
        ax_cum.grid(True, alpha=0.3)
        ax_cum.set_xlabel('Frame', fontsize=8)
        ax_cum.set_ylabel('Cumulative Error', fontsize=8)
    
    # 11. Error statistics
    if action_errors is not None:
        ax_stats = fig.add_subplot(gs[2, 3])
        ax_stats.axis('off')
        
        current_error = action_errors[frame_idx]
        total_error = np.sum(action_errors[frame_idx])
        mean_error = np.mean(action_errors[:frame_idx+1], axis=0)
        
        stats_text = f"Frame {frame_idx} Statistics:\n\n"
        stats_text += "Current Errors:\n"
        for i, label in enumerate(action_labels):
            stats_text += f"  {label}: {current_error[i]:.6f}\n"
        stats_text += f"\nTotal Error: {total_error:.6f}\n"
        stats_text += f"\nMean Errors (0-{frame_idx}):\n"
        for i, label in enumerate(action_labels):
            stats_text += f"  {label}: {mean_error[i]:.6f}\n"
        
        ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=8, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Force相关（如果有）- 全面展示所有6个维度
    if has_force:
        # 第3行：Linear Force (fx, fy, fz) + Linear Force Error
        for i, label in enumerate(['fx', 'fy', 'fz']):
            ax_f = fig.add_subplot(gs[3, i])
            ax_f.plot(gt_forces[:, i], 'b-', label='GT', linewidth=2, alpha=0.8)
            ax_f.plot(pred_forces[:, i], 'r--', label='Pred', linewidth=2, alpha=0.8)
            ax_f.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
            ax_f.set_title(f'Linear Force {label} (N)', fontsize=11, fontweight='bold')
            ax_f.legend(loc='best', fontsize=8)
            ax_f.grid(True, alpha=0.3)
            ax_f.set_xlabel('Frame', fontsize=9)
        
        # Linear Force Errors - 第3行第4列
        if force_errors is not None:
            ax_flerr = fig.add_subplot(gs[3, 3])
            for i, label in enumerate(['fx', 'fy', 'fz']):
                ax_flerr.plot(force_errors[:, i], label=label, linewidth=1.5, alpha=0.7)
            ax_flerr.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
            ax_flerr.set_title('Linear Force Errors (N)', fontsize=11, fontweight='bold')
            ax_flerr.legend(loc='best', fontsize=8)
            ax_flerr.grid(True, alpha=0.3)
            ax_flerr.set_yscale('log')
            ax_flerr.set_xlabel('Frame', fontsize=9)
        
        # 第4行：Angular Torque (mx, my, mz) + Angular Torque Error
        for i, label in enumerate(['mx', 'my', 'mz']):
            ax_t = fig.add_subplot(gs[4, i])
            ax_t.plot(gt_forces[:, i+3], 'b-', label='GT', linewidth=2, alpha=0.8)
            ax_t.plot(pred_forces[:, i+3], 'r--', label='Pred', linewidth=2, alpha=0.8)
            ax_t.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
            ax_t.set_title(f'Angular Torque {label} (Nm)', fontsize=11, fontweight='bold')
            ax_t.legend(loc='best', fontsize=8)
            ax_t.grid(True, alpha=0.3)
            ax_t.set_xlabel('Frame', fontsize=9)
        
        # Angular Torque Errors - 第4行第4列
        if force_errors is not None:
            ax_terr = fig.add_subplot(gs[4, 3])
            for i, label in enumerate(['mx', 'my', 'mz']):
                ax_terr.plot(force_errors[:, i+3], label=label, linewidth=1.5, alpha=0.7)
            ax_terr.axvline(x=frame_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
            ax_terr.set_title('Angular Torque Errors (Nm)', fontsize=11, fontweight='bold')
            ax_terr.legend(loc='best', fontsize=8)
            ax_terr.grid(True, alpha=0.3)
            ax_terr.set_yscale('log')
            ax_terr.set_xlabel('Frame', fontsize=9)
    
    # 转换为图像
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    frame = np.asarray(buf)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    return frame


def create_video(episode_data, results, output_path, fps=10):
    """创建视频（Matplotlib版本，内容全面）"""
    print(f"\n生成视频: {output_path}")
    
    has_force = 'predicted_forces' in results
    num_frames = len(episode_data['camera0_rgb'])
    
    # 准备数据
    pred_actions = results['predicted_actions']
    gt_actions = results['ground_truth_actions']
    
    # 预计算误差
    print("预计算误差...")
    action_errors = np.abs(pred_actions - gt_actions)
    cum_errors = np.cumsum(action_errors, axis=0)
    
    pred_forces = None
    gt_forces = None
    force_errors = None
    if has_force:
        pred_forces = results['predicted_forces']
        gt_forces = results.get('observed_forces', results['predicted_forces'])
        force_errors = np.abs(pred_forces - gt_forces)
    
    # 创建第一帧获取尺寸
    print("生成第一帧...")
    first_frame = create_visualization_frame(
        episode_data['camera0_rgb'][0],
        pred_actions, gt_actions, 0,
        pred_forces, gt_forces,
        action_errors, cum_errors, force_errors
    )
    
    height, width = first_frame.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    video_writer.write(first_frame)
    
    # 生成并写入所有帧
    print("渲染视频帧...")
    for t in tqdm(range(1, num_frames), desc="生成视频", ncols=80):
        frame = create_visualization_frame(
            episode_data['camera0_rgb'][t],
            pred_actions, gt_actions, t,
            pred_forces, gt_forces,
            action_errors, cum_errors, force_errors
        )
        video_writer.write(frame)
    
    video_writer.release()
    print(f"✓ 视频已保存: {output_path}")
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"  帧数: {num_frames}")
    print(f"  Action MAE: {action_errors.mean():.6f}")
    print(f"  Action RMSE: {np.sqrt((action_errors**2).mean()):.6f}")
    print(f"  Per-dimension MAE:")
    action_labels = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
    for i, label in enumerate(action_labels):
        print(f"    {label}: {action_errors[:, i].mean():.6f}")
    
    if has_force:
        print(f"\n  Force MAE: {force_errors.mean():.6f}")
        print(f"  Force RMSE: {np.sqrt((force_errors**2).mean()):.6f}")
        print(f"  Per-dimension Force MAE:")
        force_labels = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
        for i, label in enumerate(force_labels):
            print(f"    {label}: {force_errors[:, i].mean():.6f}")


@click.command()
@click.option('--checkpoint', '-c', required=True, help='模型checkpoint路径')
@click.option('--dataset-path', '-d', required=True, help='数据集路径')
@click.option('--episode-idx', '-e', default=0, type=int, help='要测试的episode索引')
@click.option('--output-path', '-o', default='open_loop_test.mp4', help='输出视频路径')
@click.option('--fps', default=10, type=int, help='视频帧率')
@click.option('--action-chunk-size', default=8, type=int, help='Action chunking大小（推理一次执行多少步）')
def main(checkpoint, dataset_path, episode_idx, output_path, fps, action_chunk_size):
    """
    开环评估：测试模型在完整episode上的表现
    
    特点：
    - Action chunking: 推理一次执行多步（更接近真实部署）
    - 绝对轨迹对比: 累加delta得到绝对轨迹
    - 优化视频生成: 重用figure加速渲染
    
    例如：
        python open_loop_evaluation.py -c checkpoints/latest.ckpt -d data/dataset.zarr.zip -e 0
        python open_loop_evaluation.py -c checkpoints/latest.ckpt -d data/dataset.zarr.zip -e 0 --action-chunk-size 8
    """
    print("=" * 80)
    print("开环评估工具 (Action Chunking版本)")
    print("=" * 80)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    policy, cfg, workspace = load_checkpoint(checkpoint)
    policy = policy.to(device)
    
    # 加载episode
    episode_data = load_episode_from_zarr(dataset_path, episode_idx)
    
    # 运行预测（使用action chunking）
    results = run_open_loop_prediction(
        policy, episode_data, cfg, 
        device=device, 
        action_chunk_size=action_chunk_size
    )
    
    # 创建视频
    create_video(episode_data, results, output_path, fps=fps)
    
    print("\n" + "=" * 80)
    print("✓ 完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

