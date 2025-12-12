# 数据转换说明

## 概述

`convert_session_to_zarr.py` 脚本用于将HDF5格式的session数据转换为UMI数据集所需的zarr格式。

## HDF5数据格式

每个episode文件包含以下数据：
- `action`: (T, 7) - 动作数据 [x, y, z, rx, ry, rz, gripper]
- `state`: (T, 7) - 状态数据 [x, y, z, rx, ry, rz, gripper]
- `image`: (T, 480, 640, 3) - RGB图像
- `force`: (T, 6) - 力/力矩传感器数据
- `timestamp*`: 各种时间戳

## 输出zarr格式

转换后的zarr文件包含：
- `data/camera0_rgb`: RGB图像 (T, H, W, 3)
- `data/robot0_eef_pos`: 末端执行器位置 (T, 3)
- `data/robot0_eef_rot_axis_angle`: 末端执行器旋转 (T, 3)
- `data/robot0_gripper_width`: 夹爪宽度 (T, 1)
- `data/action`: 动作 (T, 7)
- `meta/episode_ends`: 每个episode的结束索引

## 使用方法

### 基本用法

```bash
# 转换整个session，输出到默认位置 (session目录下的dataset.zarr.zip)
python convert_session_to_zarr.py -i data/session_20251025_142256

# 指定输出路径
python convert_session_to_zarr.py -i data/session_20251025_142256 -o my_dataset.zarr.zip
```

### 高级选项

```bash
# 自定义图像大小 (默认: 224x224)
python convert_session_to_zarr.py -i data/session_20251025_142256 -s 256x256

# 只转换前N个episodes (用于测试)
python convert_session_to_zarr.py -i data/session_20251025_142256 -n 10

# 组合使用
python convert_session_to_zarr.py -i data/session_20251025_142256 \
    -o output/my_dataset.zarr.zip \
    -s 224x224 \
    -n 50
```

### 查看帮助

```bash
python convert_session_to_zarr.py --help
```

## 在训练配置中使用

### 方法1: 使用FADP数据集（推荐，仅RGB+Action）

转换完成后，使用专门的FADP配置文件进行训练：

```bash
# 使用默认配置训练
python train.py --config-name=train_diffusion_unet_timm_fadp_workspace

# 或指定数据集路径
python train.py --config-name=train_diffusion_unet_timm_fadp_workspace \
    task.dataset_path=data/session_20251025_142256/dataset.zarr.zip
```

**FADP数据集特点:**
- 只输出RGB图像作为观测，不包含state观测（纯视觉输入）
- 使用state数据计算相对action，但不作为观测输出
- 更简单的数据处理流程，适用于纯视觉-动作学习任务
- Action格式: [x, y, z, rx, ry, rz, gripper] (7维，相对pose，axis-angle表示)
- 相对action：action相对于当前机器人末端位姿计算
- 数据增强：训练时对位姿添加高斯噪声（标准差0.05），提高模型鲁棒性

### 方法2: 使用UMI数据集（需要完整state）

如果你的数据包含完整的state信息，可以使用UMI配置：

```bash
python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset_path=data/session_20251025_142256/dataset.zarr.zip
```

**注意:** UMI数据集需要额外的低维观测数据（robot0_eef_pos、robot0_eef_rot_axis_angle等），当前转换脚本已经提供了这些数据。

## 注意事项

1. **图像尺寸**: 默认将图像缩放到224x224，这是ViT模型的标准输入尺寸。如果使用其他模型，可能需要调整。

2. **内存占用**: 转换过程会将所有数据加载到内存中。对于大型数据集，请确保有足够的RAM。

3. **压缩**: 图像使用高压缩率（'disk'模式），其他数据使用默认压缩。这可以显著减小文件大小。

4. **坐标系**: 脚本假设HDF5中的state和action使用相同的坐标系和旋转表示。如果需要转换，请修改脚本。

5. **数据增强（FADP）**: 训练时会在位姿上添加噪声，可以为每个维度单独指定标准差。配置格式：
   ```yaml
   # 方式1: 使用单个值，所有6个维度使用相同标准差
   pose_noise_scale: 0.05
   
   # 方式2: 使用列表，分别为 [x, y, z, rx, ry, rz] 指定标准差
   pose_noise_scale: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
   
   # 方式3: 不同维度使用不同的噪声强度
   pose_noise_scale: [0.01, 0.01, 0.01, 0.05, 0.05, 0.05]  # 位置噪声小，旋转噪声大
   ```
   
   通过命令行调整：
   ```bash
   # 不使用噪声
   python train.py --config-name=train_diffusion_unet_timm_fadp_workspace \
       task.dataset.pose_noise_scale=0.0
   
   # 使用单个值（所有维度相同）
   python train.py --config-name=train_diffusion_unet_timm_fadp_workspace \
       task.dataset.pose_noise_scale=0.1
   
   # 使用列表（需要特殊语法）
   python train.py --config-name=train_diffusion_unet_timm_fadp_workspace \
       task.dataset.pose_noise_scale=[0.01,0.01,0.01,0.05,0.05,0.05]
   ```
   注意：验证集始终不使用噪声。

## 验证转换结果

### 使用inspect_zarr.py脚本（推荐）

转换完成后，可以使用提供的脚本快速查看zarr文件内容：

```bash
# 基本信息查看
python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip

# 查看详细统计信息
python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --detailed

# 查看特定episode的数据
python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --episode 0

# 可视化episode长度分布
python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --visualize

# 显示某一步的图像（需要matplotlib）
python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --show-image 0

# 查看帮助
python inspect_zarr.py --help
```

### 使用Python代码验证

也可以直接使用Python代码验证数据：

```python
import zarr
import numpy as np

# 打开zarr文件
with zarr.ZipStore('data/session_20251025_142256/dataset.zarr.zip', mode='r') as store:
    root = zarr.group(store)
    
    # 查看结构
    print(root.tree())
    
    # 检查episode数量
    episode_ends = root['meta']['episode_ends'][:]
    print(f"Episodes: {len(episode_ends)}")
    print(f"Total steps: {episode_ends[-1]}")
    
    # 检查数据形状
    for key in root['data'].keys():
        data = root['data'][key]
        print(f"{key}: {data.shape}, {data.dtype}")
```

## 故障排除

如果遇到问题：

1. **ModuleNotFoundError**: 确保已安装所有依赖
   ```bash
   conda activate umi
   ```

2. **内存不足**: 使用 `-n` 参数减少转换的episodes数量

3. **文件损坏**: 脚本会跳过无法读取的episode文件并显示警告

