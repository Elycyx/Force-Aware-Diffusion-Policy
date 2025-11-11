# FADP Force Encoder 使用指南

## 概述

FADP Force Encoder 是一个融合视觉和力传感器信息的编码器，用于机器人操控任务。

### 架构特点

1. **DINOv3 ViT-B/16**: 处理RGB图像 → CLS token
2. **Force MLP Encoder**: 处理6维力传感器数据 → force embedding  
3. **Bidirectional Cross Attention**: 融合视觉和力信息
4. **输出**: 拼接后的特征向量 [cls_token, force_token]

参考论文: [DINOv3](https://github.com/facebookresearch/dinov3)

## 数据集要求

数据集需要包含以下观测：
- **camera0_rgb**: RGB图像 (T, H, W, 3)
- **force**: 力传感器数据 (T, 6) - [fx, fy, fz, mx, my, mz]
- **robot0_eef_pos**: 末端位置 (T, 3) - 用于计算相对action
- **robot0_eef_rot_axis_angle**: 末端旋转 (T, 3) - 用于计算相对action
- **robot0_gripper_width**: 夹爪宽度 (T, 1)
- **action**: 动作 (T, 7) - [x, y, z, rx, ry, rz, gripper]

## 相对表示（Relative Representation）

### Action相对表示

与UMI数据集一致，FADP使用**相对action表示**：
- **绝对action**: 目标末端位姿的世界坐标
- **相对action**: 相对于当前末端位姿的增量
- **优势**: 训练更稳定、泛化性更好、减少对初始位姿的依赖

### Force相对表示（新增）

**重要更新**：从当前版本开始，模型预测的力也采用**相对表示**：

```python
# 训练时
current_force = obs_dict['force'][-1]  # 当前观测的最后一个force (6,)
future_force_abs = data['force']       # 未来的绝对force (T, 6)
future_force_delta = future_force_abs - current_force  # 相对force (T, 6)

# 模型输入/输出
model_output = [action_delta (7,), force_delta (6,)]  # 13维

# 推理时
force_delta = model.predict(obs)       # 模型预测相对force
current_force = obs['force'][-1]       # 当前观测force
future_force_abs = force_delta + current_force  # 转换为绝对force
```

**相对力预测的优势**：
1. **与action一致**: action和force都使用相对表示，训练更统一
2. **更易学习**: 力的变化范围通常比绝对值小，梯度更稳定
3. **关注变化**: 更关注力的变化趋势（接触检测、力控制）
4. **减少误差**: 减少对初始力标定误差的依赖

**推理输出**：
- `result['action_only']`: 相对action (B, T, 7)
- `result['force_delta']`: 相对force/力增量 (B, T, 6)
- `result['force_pred']`: 绝对force（自动转换） (B, T, 6)
- `result['action_pred']`: 完整输出 (B, T, 13) - 包含action_only和force_delta

## 安装依赖

### 安装 HuggingFace Transformers（推荐）

```bash
# 安装 transformers 库用于加载 DINOv3
pip install transformers
```

DINOv3 模型会自动从 HuggingFace Hub 下载并缓存。首次使用时需要网络连接。

### 可用的 DINOv3 模型

**ViT 模型**（Vision Transformer）:
- `facebook/dinov3-vit-small-pretrain-lvd1689m` (384维)
- `facebook/dinov3-vit-base-pretrain-lvd1689m` (768维, **默认推荐**)
- `facebook/dinov3-vit-large-pretrain-lvd1689m` (1024维)
- `facebook/dinov3-vit-giant-pretrain-lvd1689m` (1536维)

**ConvNeXt 模型**:
- `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
- `facebook/dinov3-convnext-small-pretrain-lvd1689m`
- `facebook/dinov3-convnext-base-pretrain-lvd1689m`
- `facebook/dinov3-convnext-large-pretrain-lvd1689m`

### 离线使用

如果需要在无网络环境使用，可以提前下载模型：

```python
from transformers import AutoImageProcessor, AutoModel

model_name = 'facebook/dinov3-vit-base-pretrain-lvd1689m'

# 下载并缓存模型
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 模型会保存在 ~/.cache/huggingface/
```

## 配置文件

### Task 配置: `fadp_force.yaml`

定义了数据集结构和观测shape：

```yaml
shape_meta:
  obs:
    camera0_rgb:  # RGB图像
      shape: [3, 224, 224]
      type: rgb
      ignore_by_policy: False
    
    force:  # 力传感器（注意key名称）
      shape: [6]
      type: low_dim
      ignore_by_policy: False
    
    # 以下用于相对action计算，不输出给policy
    robot0_eef_pos:
      shape: [3]
      ignore_by_policy: True
    robot0_eef_rot_axis_angle:
      shape: [3]
      ignore_by_policy: True
```

### Workspace 配置: `train_diffusion_unet_fadp_force_workspace.yaml`

定义了模型架构和训练参数：

```yaml
policy:
  _target_: diffusion_policy.policy.diffusion_unet_fadp_policy.DiffusionUnetFADPPolicy
  
  obs_encoder:
    _target_: diffusion_policy.model.vision.fadp_encoder.FADPEncoder
    
    # DINOv3 模型选择 (HuggingFace格式)
    dinov3_model: 'facebook/dinov3-vit-base-pretrain-lvd1689m'
    dinov3_frozen: False  # 是否冻结DINOv3权重
    use_huggingface: True  # 使用HuggingFace transformers（推荐）
    
    # Force MLP设置
    force_mlp_hidden_dim: 256
    force_mlp_layers: 3
    
    # Cross attention设置
    num_attention_heads: 8
    attention_dropout: 0.1
    
    # 数据增强（仅在use_huggingface=False时生效）
    transforms: null  # HuggingFace使用自己的image processor
```

## 独立的历史窗口配置

**新特性**: 视觉和力传感器可以使用不同的历史窗口大小！

```yaml
img_obs_horizon: 2      # RGB图像历史窗口（默认2帧）
force_obs_horizon: 6    # 力传感器历史窗口（默认6帧）
```

**为什么分开配置？**
- **视觉**: DINOv3特征丰富，少量帧即可（2帧）
- **力**: 需要更长历史捕捉力的变化趋势（6帧）

**详细说明**: 查看 [FADP_HORIZON_CONFIG.md](FADP_HORIZON_CONFIG.md)

## 使用方法

### 1. 准备数据集

使用 `convert_session_to_zarr.py` 转换HDF5数据：

```bash
python convert_session_to_zarr.py -i data/session_20251025_142256
```

确保HDF5文件包含 `force` 数据集。如果缺失，转换脚本会自动填充零数组。

### 2. 训练模型

```bash
# 基本训练（使用默认horizon: 视觉2帧，力6帧）
python train.py --config-name=train_diffusion_unet_fadp_force_workspace

# 指定数据集路径
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    task.dataset_path=data/your_dataset/dataset.zarr.zip

# 自定义历史窗口
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    task.img_obs_horizon=4 \
    task.force_obs_horizon=8

# 使用本地DINOv3
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    policy.obs_encoder.dinov3_repo_dir=/path/to/dinov3 \
    policy.obs_encoder.dinov3_weights=/path/to/checkpoint.pth

# 冻结DINOv3权重（只训练MLP和attention）
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    policy.obs_encoder.dinov3_frozen=True
```

### 3. 选择不同的DINOv3模型

```bash
# 使用Small模型（更快，较小）
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    policy.obs_encoder.dinov3_model=dinov3_vits16

# 使用Large模型（更强，较慢）
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    policy.obs_encoder.dinov3_model=dinov3_vitl16

# 使用Huge模型（最强，最慢）
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    policy.obs_encoder.dinov3_model=dinov3_vith16plus
```

### 4. 调整Force MLP

```bash
# 更大的hidden dimension
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    policy.obs_encoder.force_mlp_hidden_dim=512

# 更多层数
python train.py --config-name=train_diffusion_unet_fadp_force_workspace \
    policy.obs_encoder.force_mlp_layers=4
```

### 5. 测试模型

```bash
# 测试验证集
python test_fadp_model.py -c data/outputs/.../latest.ckpt -v

# 测试训练集
python test_fadp_model.py -c data/outputs/.../latest.ckpt --use-train-set -v
```

## 模型特点对比

| 配置 | 编码器 | 输入 | 输出特征维度 | 适用场景 |
|------|--------|------|-------------|---------|
| **fadp.yaml** | TimmObsEncoder | RGB only | ~512-1024 | 纯视觉任务 |
| **fadp_force.yaml** | FADPEncoder | RGB + Force | 1536 (768×2) | 需要力感知的任务 |

## DINOv3 模型规格

| 模型 | 参数量 | Feature Dim | Patch Size | 推荐用途 |
|------|--------|-------------|------------|---------|
| dinov3_vits16 | 22M | 384 | 16 | 快速原型 |
| dinov3_vitb16 | 86M | 768 | 16 | **默认推荐** |
| dinov3_vitl16 | 307M | 1024 | 16 | 高精度任务 |
| dinov3_vith16plus | 632M | 1280 | 16 | 最高精度 |
| dinov3_vit7b16 | 7B | 1536 | 16 | 研究用途 |

## 架构细节

### 1. 多时间步处理流程

#### RGB图像编码（DINOv3）

```
Input: (B, T, C, H, W)  # T个时间步的RGB图像
    ↓
Reshape: (B×T, C, H, W)  # 每个时间步独立处理
    ↓
DINOv3 Forward: 每个时间步通过ViT
    ↓
CLS Token: (B×T, 768)  # 每个时间步的CLS token
    ↓
Reshape: (B, T, 768)  # 恢复batch和时间维度
    ↓
Time Average: (B, 768)  # 对所有时间步取平均
    ↓
Output: (B, 768) visual feature
```

**关键点**: 每个时间步的RGB图像独立通过DINOv3编码，然后对时间维度取平均，得到单一的特征向量。

#### Force传感器编码（MLP）

```
Input: (B, T_force, 6)  # T_force个时间步的force数据（默认T_force=6）
    ↓
Flatten: (B, T_force×6)  # 将所有时间步flatten成一维（默认36维）
    ↓
MLP: Linear(T_force×6, 256) → LayerNorm → ReLU → Dropout
    → Linear(256, 256) → LayerNorm → ReLU → Dropout
    → Linear(256, 768) → LayerNorm
    ↓
Output: (B, 768) force embedding
```

**关键点**: 
- 所有时间步的force数据flatten后一起通过MLP，MLP学习时间序列的模式
- **T_force 可独立配置**（默认6），不同于视觉的 T_img（默认2）
- MLP输入维度自动调整为 T_force × 6

### 2. Bidirectional Cross Attention

```
Visual Feature (B, 768)  ←→  Force Feature (B, 768)
        ↓                        ↓
    Visual→Force Attention    Force→Visual Attention
        ↓                        ↓
    Feed Forward            Feed Forward
        ↓                        ↓
Enhanced Visual (B, 768)    Enhanced Force (B, 768)
                ↓
        Concatenate: (B, 1536)
```

**关键点**: Cross attention在单时间步特征上进行，融合视觉和力信息。

### 3. 完整数据流

```
Obs Dict (默认配置):
  camera0_rgb: (B, T_img=2, 3, 224, 224)
  force: (B, T_force=6, 6)
        ↓
┌─────────────────────────────────────────────┐
│  DINOv3 Processing (per timestep)           │
│  (B×T_img, 3, 224, 224) → (B×T_img, 768)   │
│  → Reshape → (B, T_img, 768)                │
│  → Time Average → (B, 768)                 │
└─────────────────────────────────────────────┘
        ↓
  Visual Feature (B, 768)
        ↓
┌─────────────────────────────────────────────┐
│  Force MLP Processing                       │
│  (B, T_force, 6) → Flatten → (B, T_force×6) │
│  (B, 36) → MLP → (B, 768)                   │
└─────────────────────────────────────────────┘
        ↓
  Force Feature (B, 768)
        ↓
┌─────────────────────────────────────────────┐
│  Bidirectional Cross Attention              │
│  Visual (B, 768) ←→ Force (B, 768)          │
│  → Enhanced Visual (B, 768)                 │
│  → Enhanced Force (B, 768)                  │
└─────────────────────────────────────────────┘
        ↓
  Concatenate: (B, 1536)
        ↓
  Global Condition for UNet
```

**灵活性**: T_img 和 T_force 可独立配置，满足不同任务需求

### 4. 时间步处理策略对比

| 组件 | 输入形状 | 处理方式 | 输出形状 | 说明 |
|------|---------|---------|---------|------|
| **DINOv3** | (B, T, C, H, W) | 每个时间步独立编码 → 时间平均 | (B, 768) | 保留每个时间步的视觉信息，最后聚合 |
| **Force MLP** | (B, T, 6) | Flatten所有时间步 → MLP | (B, 768) | MLP学习时间序列模式 |
| **Cross Attention** | (B, 768) × 2 | 单时间步特征融合 | (B, 1536) | 在聚合后的特征上进行 |

**设计理由**:
- **RGB时间平均**: DINOv3已经提取了每个时间步的丰富视觉特征，平均可以平滑噪声并保留主要信息
- **Force Flatten**: Force数据维度小，flatten后MLP可以学习时间序列的完整模式（如力的变化趋势）
- **Cross Attention**: 在聚合后的特征上进行，计算效率高且能有效融合多模态信息

### 5. 与UNet的连接

```
Obs Dict {RGB, Force}
        ↓
FADPEncoder: (B, 1536) global condition
        ↓
Conditional UNet1D
        ↓
Action: (B, T, 7)
```

## 常见问题

### Q1: 如何确认force数据被正确加载？

在训练开始时，查看日志：
```
RGB keys:   ['camera0_rgb']
Force keys: ['force']
Force dimension: 6
```

### Q2: 数据集没有force数据怎么办？

`convert_session_to_zarr.py` 会自动填充零数组。但这样训练出的模型无法利用力信息，建议：
1. 收集带force的新数据
2. 或使用不带force的 `fadp.yaml` 配置

### Q3: DINOv3加载失败？

```python
RuntimeError: Failed to load DINOv3 model 'dinov3_vitb16'
```

解决方案：
1. 检查网络连接（torch hub需要下载）
2. 或使用本地加载：
```bash
git clone https://github.com/facebookresearch/dinov3.git
python train.py ... policy.obs_encoder.dinov3_repo_dir=/path/to/dinov3
```

### Q4: 内存不足？

1. 减小batch size:
```bash
python train.py ... dataloader.batch_size=32
```

2. 使用smaller模型:
```bash
python train.py ... policy.obs_encoder.dinov3_model=dinov3_vits16
```

3. 冻结DINOv3:
```bash
python train.py ... policy.obs_encoder.dinov3_frozen=True
```

### Q5: 训练速度慢？

1. 冻结DINOv3（只训练MLP和attention）:
```bash
policy.obs_encoder.dinov3_frozen=True
```

2. 减少数据增强:
```yaml
transforms: []  # 移除所有transforms
```

3. 使用FP16混合精度训练（需要修改训练脚本）

## 性能优化建议

1. **初始训练**: 冻结DINOv3，只训练MLP和attention（几个epoch）
2. **Fine-tuning**: 解冻DINOv3，使用更小的学习率继续训练
3. **数据增强**: 根据任务调整ColorJitter参数
4. **Force归一化**: 根据实际force范围调整归一化参数

## 文件清单

```
diffusion_policy/
├── model/vision/
│   └── fadp_encoder.py          # FADP编码器实现
├── policy/
│   └── diffusion_unet_fadp_policy.py  # FADP策略
├── config/
│   ├── task/
│   │   ├── fadp.yaml            # 纯视觉配置
│   │   └── fadp_force.yaml      # 视觉+力配置
│   └── train_diffusion_unet_fadp_force_workspace.yaml
```

## 参考资料

- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [DINOv3 Paper](https://arxiv.org/abs/2508.10104)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)

