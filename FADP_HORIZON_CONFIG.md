# FADP 历史窗口配置指南

## 概述

FADP Encoder 支持为视觉（RGB）和力传感器配置**独立的历史窗口大小**，以便更好地捕捉不同模态的时序信息。

## 默认配置

```yaml
img_obs_horizon: 2      # RGB图像历史窗口 = 2帧
force_obs_horizon: 6    # 力传感器历史窗口 = 6帧
```

**设计理由**:
- **视觉 (2帧)**: DINOv3提取的特征已经非常丰富，少量帧即可捕捉场景信息
- **力传感器 (6帧)**: 力的变化趋势需要更长的历史来捕捉（如接触检测、力控制）

## 配置方法

### 方式1: 修改配置文件

编辑 `diffusion_policy/config/task/fadp_force.yaml`:

```yaml
# 观测和动作的时间窗口
img_obs_horizon: 2      # 修改这里改变视觉历史窗口
force_obs_horizon: 6    # 修改这里改变力传感器历史窗口
action_horizon: 16
obs_down_sample_steps: 3
```

### 方式2: 命令行参数

```bash
# 使用默认值训练
python train.py --config-name=train_diffusion_unet_fadp_force_workspace

# 自定义视觉和力的历史窗口
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    task.img_obs_horizon=4 \
    task.force_obs_horizon=8

# 使用相同的历史窗口
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    task.img_obs_horizon=4 \
    task.force_obs_horizon=4
```

## 不同任务的推荐配置

### 1. 精细操作任务（Assembly, Insertion）

```yaml
img_obs_horizon: 1      # 实时视觉反馈
force_obs_horizon: 8    # 长历史捕捉力的变化
```

**理由**: 精细操作需要快速视觉反馈，但力控制需要更长的历史来稳定。

### 2. 动态抓取任务（Pick & Place）

```yaml
img_obs_horizon: 3      # 捕捉物体运动
force_obs_horizon: 4    # 中等力历史
```

**理由**: 需要追踪移动物体，但力变化相对简单。

### 3. 接触丰富任务（Wiping, Pushing）

```yaml
img_obs_horizon: 2      # 标准视觉
force_obs_horizon: 10   # 长力历史
```

**理由**: 接触任务高度依赖力反馈，需要长历史捕捉力的模式。

### 4. 静态场景任务（Button Pressing）

```yaml
img_obs_horizon: 1      # 单帧足够
force_obs_horizon: 6    # 标准力历史
```

**理由**: 场景静态，单帧视觉即可，力用于检测接触。

### 5. 高速任务（Fast Manipulation）

```yaml
img_obs_horizon: 4      # 更多帧捕捉快速运动
force_obs_horizon: 8    # 更多帧捕捉快速力变化
```

**理由**: 高速运动需要更多历史来预测未来状态。

## 数据流详解

### 当前配置 (视觉2帧，力6帧)

```
时间轴: t₀  t₁  t₂  t₃  t₄  t₅  t₆  (当前时刻)
        │   │   │   │   │   │   │
视觉:    ─   ─   ─   ─   ─   ●───●   (使用 t₅, t₆)
        │   │   │   │   │   │   │
力:      ─   ●───●───●───●───●───●   (使用 t₁, t₂, t₃, t₄, t₅, t₆)
                ↓
        RGB: (B, 2, 3, 224, 224)
        Force: (B, 6, 6)
                ↓
        DINOv3 处理每个时间步 → 平均
        Force MLP flatten 所有时间步
                ↓
        Visual Feature: (B, 768)
        Force Feature: (B, 768)
                ↓
        Cross Attention 融合
                ↓
        Output: (B, 1536)
```

### 自定义配置示例 (视觉4帧，力8帧)

```
时间轴: t₀  t₁  t₂  t₃  t₄  t₅  t₆  t₇  t₈  (当前时刻)
        │   │   │   │   │   │   │   │   │
视觉:    ─   ─   ─   ─   ─   ●───●───●───●   (使用 t₅, t₆, t₇, t₈)
        │   │   │   │   │   │   │   │   │
力:      ─   ●───●───●───●───●───●───●───●   (使用 t₁~t₈)
                ↓
        RGB: (B, 4, 3, 224, 224)
        Force: (B, 8, 6)
                ↓
        Force MLP input: (B, 48) = 8×6
```

## 内存和计算成本

### 内存消耗

| 视觉Horizon | 力Horizon | DINOv3输入 | Force MLP输入 | 总增长 |
|------------|----------|-----------|--------------|--------|
| 2          | 6        | (B×2, 3, 224, 224) | (B, 36) | 基线 |
| 4          | 6        | (B×4, 3, 224, 224) | (B, 36) | 2× 视觉 |
| 2          | 12       | (B×2, 3, 224, 224) | (B, 72) | 2× 力 |
| 4          | 12       | (B×4, 3, 224, 224) | (B, 72) | 2× 视觉 + 2× 力 |

**关键点**:
- **视觉horizon**: 线性影响DINOv3计算时间和GPU内存
- **力horizon**: 线性影响Force MLP输入维度和参数量（但总体较小）

### 计算时间估计

假设 batch_size=64, 单帧DINOv3耗时 10ms:

```python
# 视觉2帧 + 力6帧 (默认)
DINOv3_time = 2 * 10ms = 20ms
Force_MLP_time ≈ 1ms
Total ≈ 21ms/batch

# 视觉4帧 + 力12帧
DINOv3_time = 4 * 10ms = 40ms
Force_MLP_time ≈ 2ms
Total ≈ 42ms/batch  (约2倍)

# 视觉1帧 + 力6帧 (最快)
DINOv3_time = 1 * 10ms = 10ms
Force_MLP_time ≈ 1ms
Total ≈ 11ms/batch  (约0.5倍)
```

## 数据集要求

### 最小Episode长度

```python
min_episode_length = max(
    img_obs_horizon,
    force_obs_horizon
) + action_horizon

# 默认配置
min_length = max(2, 6) + 16 = 22 steps

# 自定义配置示例 (4, 8, 16)
min_length = max(4, 8) + 16 = 24 steps
```

**检查方法**:
```bash
# 使用 inspect_zarr.py 检查数据集
python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --detailed

# 查看最短episode长度
# 确保 min_episode_length >= 所需长度
```

## 调试和验证

### 训练时的日志输出

启动训练后，查看日志：

```
RGB keys: ['camera0_rgb'], horizon: 2
Force keys: ['force'], horizon: 6
Force dimension: 6
Force MLP input dimension: 36 = 6 (horizon) × 6 (force_dim)
DINOv3 feature dimension: 768
Total parameters: 89,234,567
Trainable parameters: 89,234,567
```

### 验证配置是否生效

```python
# 在训练脚本中添加
print(f"Image obs horizon: {cfg.task.img_obs_horizon}")
print(f"Force obs horizon: {cfg.task.force_obs_horizon}")

# 或在dataset中检查
train_dataset = ...
sample = train_dataset[0]
print(f"RGB shape: {sample['obs']['camera0_rgb'].shape}")  # 应为 (2, 3, 224, 224)
print(f"Force shape: {sample['obs']['force'].shape}")      # 应为 (6, 6)
```

## 常见问题

### Q1: 为什么力的默认horizon (6) 比视觉 (2) 大？

**A**: 
- **视觉**: DINOv3提取的特征已经包含丰富的语义信息，少量帧足够
- **力**: 原始6D力/力矩信号，需要更长历史来捕捉变化趋势（如接触瞬间、力控制稳定性）

### Q2: 增加horizon会提升性能吗？

**A**: 不一定。需要根据任务：
- **静态任务**: 更多历史无益，可能引入噪声
- **动态任务**: 适度增加可捕捉运动模式
- **过长历史**: 可能导致过拟合，训练困难

建议从默认值开始，根据验证集性能调整。

### Q3: 两者可以相等吗？

**A**: 可以。设置为相同值：
```bash
python train.py ... task.img_obs_horizon=4 task.force_obs_horizon=4
```

### Q4: 最大可以设置多少？

**A**: 受限于：
1. **数据集最短episode长度**
2. **GPU内存**（主要是视觉horizon）
3. **训练稳定性**（过长难以学习）

实践中，视觉horizon < 8，力horizon < 16 比较合理。

### Q5: 训练中可以动态调整吗？

**A**: 不建议。Horizon影响模型输入维度：
- 视觉: 只影响时间平均，可以一定程度变化
- 力: 直接影响MLP输入维度，**不能改变**（需要重新初始化模型）

建议在训练前确定好配置。

## 高级用法

### 多摄像头不同horizon

```yaml
obs:
  camera0_rgb:
    horizon: 2  # 主相机
  camera1_rgb:
    horizon: 4  # 辅助相机，更长历史
```

当前实现会对所有RGB取平均，建议使用相同horizon。

### 自适应horizon (TODO)

未来可以实现：
```python
# 根据任务阶段动态调整
if contact_detected:
    force_horizon = 10  # 接触时增加
else:
    force_horizon = 4   # 非接触时减少
```

## 总结

| 参数 | 默认值 | 推荐范围 | 影响 |
|------|--------|---------|------|
| **img_obs_horizon** | 2 | 1-6 | DINOv3计算时间、GPU内存 |
| **force_obs_horizon** | 6 | 4-12 | Force MLP参数量（影响较小） |

**最佳实践**:
1. 从默认值 (2, 6) 开始
2. 根据任务特点调整（参考推荐配置）
3. 监控训练/验证性能
4. 注意GPU内存和计算时间
5. 确保数据集episode足够长

