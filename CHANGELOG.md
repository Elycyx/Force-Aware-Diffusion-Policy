# Force-Aware Diffusion Policy (FADP) - 变更日志

## 核心功能添加

### 1. 力数据编码器 (ForceEncoder)
**文件**: `fadp/model/common/force_encoder.py`

新增了专门处理力/力矩传感器数据的MLP编码器：
- 输入: 6维力/力矩测量值 (fx, fy, fz, mx, my, mz)
- 架构: Linear(6 → 128) → Swish → Linear(128 → output_dim)
- 输出维度与RGB编码器输出维度匹配，便于特征融合

### 2. 策略模型修改
**文件**: `fadp/policy/diffusion_unet_hybrid_image_policy.py`

主要修改：
- 添加了`force_encoder`成员变量和相关初始化逻辑
- 添加`force_encoder_hidden_dim`超参数
- 在`__init__`中解析shape_meta时识别force字段
- 更新`global_cond_dim`计算：包含RGB特征 + 力特征
- 修改`predict_action`方法：
  - 分离force数据和其他观测数据
  - 使用force_encoder编码力数据
  - 将力特征与RGB特征拼接
- 修改`compute_loss`方法：同样处理force数据编码和特征拼接

关键变化：
```python
# 之前
global_cond_dim = obs_feature_dim * n_obs_steps

# 之后  
total_feature_dim = obs_feature_dim + force_feature_dim
global_cond_dim = total_feature_dim * n_obs_steps
```

### 3. 数据集更新
**文件**: `fadp/dataset/real_pusht_image_dataset.py`

修改内容：
- 添加`force_keys`列表来跟踪力数据字段
- 在数据加载时处理force字段（与lowdim_keys分开）
- 在`get_normalizer`中为force数据创建独立的归一化器
- 在`__getitem__`中返回force数据
- 在`_get_replay_buffer`中支持6维force和7维action

### 4. 配置文件更新
**文件**: `fadp/config/task/real_pusht_image.yaml`

shape_meta修改：
```yaml
obs:
  camera_1:
    shape: [3, 240, 320]
    type: rgb
  camera_3:
    shape: [3, 240, 320]
    type: rgb
  force:  # 新增
    shape: [6]
    type: low_dim
action:
  shape: [7]  # 从2改为7
```

### 5. 真机推理工具更新
**文件**: `fadp/real_world/real_inference_util.py`

- 在`get_real_obs_dict`中添加force数据处理逻辑
- 确保force数据保持6维不变

## 代码清理

### 删除的环境模块
- `fadp/env/pusht/` - PushT仿真环境
- `fadp/env/kitchen/` - Kitchen仿真环境  
- `fadp/env/block_pushing/` - Block Pushing仿真环境
- `fadp/env/robomimic/` - RoboMimic仿真环境

### 删除的数据集
- `blockpush_lowdim_dataset.py`
- `kitchen_lowdim_dataset.py`
- `kitchen_mjl_lowdim_dataset.py`
- `mujoco_image_dataset.py`
- `pusht_dataset.py`
- `pusht_image_dataset.py`
- `robomimic_replay_image_dataset.py`
- `robomimic_replay_lowdim_dataset.py`

### 删除的Runner
- `base_lowdim_runner.py`
- `blockpush_lowdim_runner.py`
- `kitchen_lowdim_runner.py`
- `pusht_image_runner.py`
- `pusht_keypoints_runner.py`
- `robomimic_image_runner.py`
- `robomimic_lowdim_runner.py`

### 删除的Policy
- `base_lowdim_policy.py`
- `bet_lowdim_policy.py`
- `diffusion_transformer_hybrid_image_policy.py`
- `diffusion_transformer_lowdim_policy.py`
- `diffusion_unet_image_policy.py`
- `diffusion_unet_lowdim_policy.py`
- `diffusion_unet_video_policy.py`
- `ibc_dfo_hybrid_image_policy.py`
- `ibc_dfo_lowdim_policy.py`
- `robomimic_image_policy.py`
- `robomimic_lowdim_policy.py`

### 删除的Workspace
- `train_bet_lowdim_workspace.py`
- `train_diffusion_transformer_*_workspace.py`
- `train_diffusion_unet_image_workspace.py`
- `train_diffusion_unet_lowdim_workspace.py`
- `train_diffusion_unet_video_workspace.py`
- `train_ibc_dfo_*_workspace.py`
- `train_robomimic_*_workspace.py`

### 删除的配置文件
除了`train_diffusion_unet_*_hybrid_workspace.yaml`外的所有训练配置
所有task配置文件除了`real_pusht_image.yaml`

### 删除的顶层脚本
- `demo_pusht.py`
- `demo_real_robot.py`
- `eval.py`
- `image_pusht_diffusion_policy_cnn.yaml`

### 删除的测试文件
- `test_block_pushing.py`
- `test_robomimic_image_runner.py`
- `test_robomimic_lowdim_runner.py`

## 保留的核心模块

### Policy & Workspace
- `fadp/policy/base_image_policy.py`
- `fadp/policy/diffusion_unet_hybrid_image_policy.py`
- `fadp/workspace/base_workspace.py`
- `fadp/workspace/train_diffusion_unet_hybrid_workspace.py`

### 数据集
- `fadp/dataset/base_dataset.py`
- `fadp/dataset/real_pusht_image_dataset.py`

### 真机相关
- `fadp/real_world/` (全部保留)
- `eval_real_robot.py`

### 模型组件
- `fadp/model/` (全部保留，新增force_encoder.py)
- `fadp/common/` (全部保留)
- `fadp/gym_util/` (保留)
- `fadp/shared_memory/` (保留)

### 工具脚本
- `train.py`
- `ray_train_multirun.py`
- `multirun_metrics.py`

## 文档更新

### 新增文档
- `FADP_README.md` - FADP详细说明
- `CHANGELOG.md` - 变更日志（本文件）
- `example_fadp_config.yaml` - 配置文件示例

### 修改文档
- `README.md` - 添加FADP说明和链接

## 架构总结

```
输入观测:
├── RGB图像 (多相机) → ResNet编码器 → RGB特征
└── 力/力矩数据 (6D) → MLP编码器 → 力特征

特征融合:
[RGB特征 | 力特征] → 全局条件 (global_cond)

扩散模型:
Conditional UNet 1D(
  condition=全局条件,
  input=噪声轨迹
) → 去噪 → 动作轨迹

输出:
7维动作 (dx, dy, dz, drx, dry, drz, gripper)
```

## 下一步工作

1. **数据收集**: 需要收集包含力传感器数据的真机演示数据
2. **硬件接口**: 修改`RealEnv`类以读取实际的力传感器数据
3. **训练验证**: 使用新数据集训练FADP模型
4. **真机测试**: 在实际机器人上测试策略性能
5. **超参数调优**: 根据任务调整force_encoder的架构和训练参数

## 兼容性说明

- 保持与原始Diffusion Policy的conda环境兼容
- 所有依赖项不变
- 训练和推理接口保持一致
- 可以通过配置文件控制是否使用force输入（在shape_meta中添加或删除force字段）

