# 使用FADP数据集训练UMI Diffusion模型 - 配置说明

## 概述

本配置允许使用FADP数据集训练UMI的diffusion模型，保持UMI的所有超参数和训练流程不变。

**关键改进**：创建了`FadpUmiDataset`类，自动将FADP的欧拉角格式转换为UMI期望的rotation vector格式。

## 创建的文件

### 1. `diffusion_policy/dataset/fadp_umi_dataset.py` ⭐ **新增**
FADP-UMI数据集适配器，解决旋转表示不兼容问题：
- **FADP存储**: 欧拉角 (Euler angles, xyz order)
- **UMI期望**: Rotation vector (axis-angle)
- **自动转换**: 在数据加载时将欧拉角转换为rotation vector

转换逻辑：
```python
# FADP: 欧拉角 [roll, pitch, yaw]
euler = np.array([rx, ry, rz])

# 转换为UMI的rotation vector
rot = st.Rotation.from_euler('xyz', euler, degrees=False)
rotvec = rot.as_rotvec()  # axis-angle表示

# UMI可以正确解析rotvec
rot_umi = st.Rotation.from_rotvec(rotvec)
```

### 2. `diffusion_policy/config/task/fadp_umi.yaml`
### 2. `diffusion_policy/config/task/fadp_umi.yaml`
Task配置文件，定义了数据集和观测/动作的shape：
- **数据集**: `data/basin_noup/dataset.zarr.zip`（FADP数据集）
- **Dataset类**: `FadpUmiDataset` ⭐ 使用新的适配器
- **观测输入**: 
  - `camera0_rgb`: RGB图像 (3, 224, 224)
  - `robot0_eef_pos`: 机器人末端位置 (3,)
  - `robot0_eef_rot_axis_angle`: 机器人末端旋转，转换为rotation_6d (6,)
  - `robot0_gripper_width`: 夹爪宽度 (1,)
- **动作输出**: 10维 (3维pos + 6维rot_6d + 1维gripper)

### 3. `diffusion_policy/config/train_diffusion_unet_timm_fadp_umi_workspace.yaml`
### 3. `diffusion_policy/config/train_diffusion_unet_timm_fadp_umi_workspace.yaml`
训练配置文件，完全使用UMI的超参数（已创建）

### 4. `test_fadp_umi_dataset.py`
验证脚本，用于测试数据加载是否正确（已创建）

### 5. `test_rotation_conversion.py` ⭐ **新增**
验证旋转转换逻辑是否正确的独立测试

## 关键技术解决方案

### 旋转表示不兼容问题

#### 问题描述
- **FADP数据集**: 使用欧拉角 (Euler angles) 表示旋转
  ```python
  # fadp_dataset.py
  rot = R.from_euler('xyz', euler[i], degrees=False)
  ```

- **UMI的pose_util**: 期望rotation vector (axis-angle)
  ```python
  # umi/common/pose_util.py
  rot = st.Rotation.from_rotvec(pose[...,3:])
  ```

#### 解决方案：FadpUmiDataset适配器

创建`FadpUmiDataset`类，在数据加载时自动转换：

```python
# 从FADP数据集读取欧拉角
euler_data = replay_buffer['robot0_eef_rot_axis_angle'][:]  # (N, 3)

# 转换为rotation vector
rotvec_data = np.zeros_like(euler_data)
for i in range(len(euler_data)):
    rot = st.Rotation.from_euler('xyz', euler_data[i], degrees=False)
    rotvec_data[i] = rot.as_rotvec()

# 替换replay_buffer中的数据
replay_buffer.data['robot0_eef_rot_axis_angle'][:] = rotvec_data
```

转换后，数据与UMI完全兼容，UmiDataset的所有后续处理（相对位姿计算、rotation_6d转换等）都能正确工作。

## 核心设计决策

### 1. 使用FadpUmiDataset而非直接使用UmiDataset
### 1. 使用FadpUmiDataset适配器
**原因**:
- 解决旋转表示不兼容：FADP用欧拉角，UMI用rotation vector
- 继承UmiDataset的所有优点（rotation_6d转换、相对位姿计算）
- 转换发生在数据加载时，一次性完成
- 对训练流程完全透明

### 2. 移除robot0_eef_rot_axis_angle_wrt_start
**原因**:
- 这个特征需要`demo_start_pose`字段，FADP数据集中没有
- 这是额外的观测特征，不是必需的
- 基本的proprioception（pos, rot, gripper）已足够

### 3. 使用obs_down_sample_steps=1
**原因**:
- FADP数据集使用采样率1
- 保持原始数据的时间分辨率
- UMI使用3，但这个参数应该根据数据集特性调整

### 4. Relative Pose计算
**FADP和UMI的共同点**:
- 都使用`pose_mat[-1]`（最后一个观测时刻）作为base_pose
- 都计算相对变换：`T_relative = T_base^(-1) * T_target`
- 不需要episode start pose

**唯一区别**:
- FADP输出3维axis_angle，UMI输出6维rotation_6d
- UmiDataset会自动转换：axis_angle → matrix → rotation_6d

## 数据转换流程

```
FADP数据集 (Zarr格式 - 欧拉角)
├── camera0_rgb: (T, H, W, 3) uint8
├── robot0_eef_pos: (T, 3) float32
├── robot0_eef_rot_axis_angle: (T, 3) float32  ← 欧拉角 (Euler angles)
├── robot0_gripper_width: (T, 1) float32
└── action: (T, 7) float32  ← [pos(3) + euler(3) + gripper(1)]

                ↓ FadpUmiDataset: 欧拉角 → rotation vector

内存中的replay_buffer (rotation vector)
├── camera0_rgb: (T, H, W, 3) uint8
├── robot0_eef_pos: (T, 3) float32
├── robot0_eef_rot_axis_angle: (T, 3) float32  ← rotation vector (axis-angle)
├── robot0_gripper_width: (T, 1) float32
└── action: (T, 7) float32  ← [pos(3) + rotvec(3) + gripper(1)]

                ↓ UmiDataset: rotation vector → rotation_6d

观测数据（输出到policy）:
├── camera0_rgb: (batch, T, 3, 224, 224) float32 [0-1]
├── robot0_eef_pos: (batch, T, 3) float32
├── robot0_eef_rot_axis_angle: (batch, T, 6) float32  ← rotation_6d
└── robot0_gripper_width: (batch, T, 1) float32

动作数据（输出到policy）:
└── action: (batch, T, 10) float32  ← [pos(3) + rot_6d(6) + gripper(1)]
```

## 旋转表示对比

| 表示法 | 维度 | FADP使用 | UMI使用 | Policy使用 | 描述 |
|--------|------|----------|---------|------------|------|
| Euler angles | 3 | ✓ | ✗ | ✗ | 欧拉角，xyz顺序 |
| Rotation vector | 3 | ✗ | ✓ | ✗ | axis-angle，UMI中间格式 |
| Rotation_6d | 6 | ✗ | ✗ | ✓ | 旋转矩阵前两行，policy输入 |

转换链：**Euler → Rotation Vector → Rotation_6d**

## 如何使用

### 1. 测试旋转转换逻辑（推荐先运行）
```bash
python test_rotation_conversion.py
```

验证欧拉角到rotation vector的转换是否正确。

### 2. 测试数据加载
```bash
python test_fadp_umi_dataset.py
```

这会验证:
- FadpUmiDataset能否正确加载FADP数据
- 旋转格式转换是否成功
- Rotation是否正确转换为6维
- 数据shape是否符合预期
- Normalizer是否工作正常

### 3. 开始训练
```bash
python train.py --config-name=train_diffusion_unet_timm_fadp_umi_workspace
```

### 3. 开始训练
```bash
python train.py --config-name=train_diffusion_unet_timm_fadp_umi_workspace
```

### 4. 监控训练
训练日志会上传到WandB项目：`fadp_umi_training`

## 预期结果

- **训练样本数**: 取决于FADP数据集大小
- **验证比例**: 5% (val_ratio=0.05)
- **Checkpoint保存**: 每10个epoch，保存top-5最佳模型
- **输出目录**: `data/outputs/<date>/<time>_train_diffusion_unet_timm_fadp_umi_fadp_umi/`

## 故障排除

### 如果遇到shape不匹配错误：
1. 运行`test_fadp_umi_dataset.py`查看具体的shape
2. 检查FADP数据集是否包含所有必需的字段
3. 确认数据集路径正确：`data/basin_noup/dataset.zarr.zip`

### 如果遇到rotation转换错误：
1. 运行`python test_rotation_conversion.py`验证转换逻辑
2. 检查FADP数据集中的值是否在合理范围（欧拉角通常在[-π, π]）
3. 确认FadpUmiDataset正确转换了所有相关字段

### 如果训练loss异常：
- 检查数据归一化是否正常（运行测试脚本验证）
- 确认学习率和batch_size设置合理
- 检查是否有NaN或Inf值

## 与原始配置的对比

| 项目 | FADP原始 | UMI原始 | 本配置 |
|------|---------|---------|--------|
| Dataset类 | FadpDataset | UmiDataset | **FadpUmiDataset**（新） ✓ |
| 旋转存储 | Euler angles | Rotation vector | Euler → RV转换 ✓ |
| Encoder | FADPEncoder (DINOv3) | TimmObsEncoder (ViT CLIP) | TimmObsEncoder ✓ |
| 观测输入 | RGB + Force | RGB + State | RGB + State ✓ |
| Action维度 | 13 (含force) | 10 | 10 ✓ |
| 旋转表示 | 3D euler | 6D rotation_6d | 6D rotation_6d ✓ |
| 采样步长 | 1 | 3 | 1 (保持FADP) |
| 学习率 | 1e-4 | 3e-4 | 3e-4 ✓ |
| Batch size | 128 | 64 | 64 ✓ |
| Epochs | 800 | 120 | 120 ✓ |

## 总结

配置已完成，所有文件已创建。核心改进：

1. ✅ **创建FadpUmiDataset适配器**：自动转换欧拉角→rotation vector
2. ✅ **使用UMI的policy和训练流程**：完全不变
3. ✅ **使用FADP的数据集**：RGB + State（无force）
4. ✅ **无缝兼容**：转换对训练流程完全透明

**转换流程**：FADP欧拉角 → FadpUmiDataset转换 → UMI rotation vector → UmiDataset处理 → rotation_6d → Policy

这样既能利用FADP的数据，又能保持UMI已经验证过的优秀训练配置，同时解决了旋转表示不兼容的关键问题！
