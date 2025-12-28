# Velocity 输入格式说明（部署时）

## 概述

当 FADP 模型配置了 `use_velocity=True` 时，需要在推理时提供 velocity 输入。Velocity 表示机器人末端执行器的相对运动速度，会被直接拼接到 encoder 输出特征中（不经过神经网络处理）。

## Velocity 定义

Velocity 是一个 7 维向量，格式为：
```
velocity = [delta_pos(3), delta_rot_euler(3), gripper_abs(1)]
```

- **delta_pos (3维)**: 位置变化量 `[dx, dy, dz]`，单位：米
- **delta_rot_euler (3维)**: 旋转变化量（欧拉角）`[droll, dpitch, dyaw]`，单位：弧度
- **gripper_abs (1维)**: 夹爪宽度绝对值，单位：米

## 输入格式

在调用 `policy.predict_action(obs_dict)` 时，`obs_dict` 中需要包含 `'velocity'` key：

### 格式 1: 单时间步 (推荐)
```python
obs_dict = {
    'camera0_rgb': ...,  # (B, T, C, H, W)
    'force': ...,        # (B, T, 6)
    'velocity': velocity_tensor  # (B, 7) - 单时间步
}
```

### 格式 2: 多时间步
```python
obs_dict = {
    'camera0_rgb': ...,  # (B, T, C, H, W)
    'force': ...,        # (B, T, 6)
    'velocity': velocity_tensor  # (B, T, 7) - 多时间步
}
```

**注意**: 如果提供的是多时间步格式 `(B, T, 7)`，模型会自动取最后一个时间步 `velocity[:, -1, :]`。

## 计算方式

Velocity 应该基于**过去两帧的机器人状态**计算相对变换：

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_velocity(prev_pose, curr_pose, curr_gripper_width):
    """
    计算 velocity
    
    Args:
        prev_pose: 上一帧的位姿 [x, y, z, rx, ry, rz] (位置 + 轴角旋转)
        curr_pose: 当前帧的位姿 [x, y, z, rx, ry, rz]
        curr_gripper_width: 当前帧的夹爪宽度 (标量)
    
    Returns:
        velocity: (7,) numpy array
    """
    # 构建 4x4 变换矩阵
    prev_mat = pose_to_mat(prev_pose)  # 需要实现 pose_to_mat 函数
    curr_mat = pose_to_mat(curr_pose)
    
    # 计算相对变换: T_{t-1}^{-1} * T_t
    rel_mat = np.linalg.inv(prev_mat) @ curr_mat
    
    # 提取位置变化
    delta_pos = rel_mat[:3, 3]  # (3,)
    
    # 提取旋转变化（转换为欧拉角）
    rel_rot = R.from_matrix(rel_mat[:3, :3])
    delta_rot_euler = rel_rot.as_euler('xyz', degrees=False)  # (3,)
    
    # 拼接
    velocity = np.concatenate([
        delta_pos,           # (3,)
        delta_rot_euler,      # (3,)
        [curr_gripper_width]  # (1,)
    ]).astype(np.float32)  # (7,)
    
    return velocity
```

## 示例代码

### 完整部署示例

```python
import torch
import numpy as np

# 假设你已经加载了模型
policy = ...  # DiffusionUnetFADPPolicy 实例

# 准备观测数据
obs_dict = {
    'camera0_rgb': rgb_images,  # (B, T, C, H, W)
    'force': force_data,        # (B, T, 6)
}

# 计算 velocity（需要维护历史状态）
if policy.obs_encoder.use_velocity:
    # 假设你有上一帧和当前帧的状态
    prev_eef_pos = ...      # (3,) 上一帧末端位置
    prev_eef_rot = ...      # (3,) 上一帧末端旋转（轴角）
    curr_eef_pos = ...      # (3,) 当前帧末端位置
    curr_eef_rot = ...      # (3,) 当前帧末端旋转（轴角）
    curr_gripper_width = ...  # 标量，当前夹爪宽度
    
    # 计算 velocity
    prev_pose = np.concatenate([prev_eef_pos, prev_eef_rot])
    curr_pose = np.concatenate([curr_eef_pos, curr_eef_rot])
    velocity = compute_velocity(prev_pose, curr_pose, curr_gripper_width)
    
    # 转换为 tensor 并添加到 obs_dict
    obs_dict['velocity'] = torch.from_numpy(velocity).unsqueeze(0)  # (1, 7)
    # 或者如果是 batch:
    # obs_dict['velocity'] = torch.from_numpy(velocity).unsqueeze(0).repeat(B, 1)  # (B, 7)

# 推理
with torch.no_grad():
    result = policy.predict_action(obs_dict)
    action = result['action']  # (B, T, 7)
```

## 注意事项

1. **如果没有提供 velocity**:
   - 如果模型配置了 `use_velocity=True` 但没有在 `obs_dict` 中提供 `'velocity'` key，模型会自动填充为全零 `(B, 7)`。
   - 这不会导致错误，但可能会影响模型性能。

2. **时间步对齐**:
   - Velocity 应该与当前观测帧对齐。
   - 如果使用多时间步格式，确保 `velocity` 的时间步数与 `camera0_rgb` 和 `force` 一致。

3. **单位一致性**:
   - 位置单位：米
   - 旋转单位：弧度（不是度）
   - 夹爪宽度单位：米

4. **历史状态维护**:
   - 部署时需要维护至少一帧的历史状态（上一帧的位姿和夹爪宽度）才能计算 velocity。
   - 第一次调用时，如果没有历史状态，可以使用全零 velocity。

## 检查模型是否使用 velocity

```python
# 检查 encoder 是否配置了 use_velocity
if hasattr(policy.obs_encoder, 'use_velocity'):
    if policy.obs_encoder.use_velocity:
        print("模型需要 velocity 输入")
        print(f"Velocity 维度: {policy.obs_encoder.velocity_dim}")  # 应该是 7
    else:
        print("模型不需要 velocity 输入")
```

