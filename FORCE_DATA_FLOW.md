# Force数据流详解

本文档详细说明力传感器数据(fx, fy, fz, mx, my, mz)如何在FADP系统中流动和处理。

## 1. 数据格式定义

### 在配置文件中定义 (`fadp/config/task/real_pusht_image.yaml`)

```yaml
shape_meta:
  obs:
    camera_1:
      shape: [3, 240, 320]  # RGB图像
      type: rgb
    force:                   # 力传感器数据
      shape: [6]             # 6维: fx, fy, fz, mx, my, mz
      type: low_dim          # 标记为低维数据类型
  action:
    shape: [7]               # 7维动作输出
```

**关键点**：
- `force`字段名称固定（代码中通过检查`key == 'force'`来识别）
- `shape: [6]`表示6维力/力矩数据
- `type: low_dim`表示这是低维连续数据（非图像）

## 2. 数据存储（训练数据）

### 数据集格式 (zarr格式)

数据应该以zarr格式存储，目录结构如下：

```
data/your_dataset/
├── data/
│   ├── camera_1/          # RGB图像数据
│   ├── force/             # 力传感器数据 ← 新增
│   └── action/            # 动作数据
└── meta/
    └── episode_ends.json  # 每个episode的结束索引
```

**force数据格式**：
- Shape: `(total_timesteps, 6)`
- 数据类型: `float32`
- 维度含义: `[fx, fy, fz, mx, my, mz]`
  - fx, fy, fz: 沿x, y, z轴的力 (单位: N)
  - mx, my, mz: 绕x, y, z轴的力矩 (单位: N·m)

### 数据加载 (`real_pusht_image_dataset.py`)

```python
# 在_get_replay_buffer函数中
replay_buffer = real_data_to_replay_buffer(
    dataset_path=dataset_path,
    out_store=store,
    out_resolutions=out_resolutions,
    lowdim_keys=lowdim_keys + ['action'],  # force会被包含在lowdim_keys中
    image_keys=rgb_keys
)
```

数据加载器会：
1. 识别shape_meta中的force字段
2. 将force添加到`force_keys`列表
3. 从zarr文件中读取force数据
4. 应用归一化

## 3. 训练时的数据流

### Step 1: 数据采样 (`__getitem__`)

```python
# 在RealPushTImageDataset.__getitem__中
obs_dict = {
    'camera_1': np.array([...]),  # Shape: (n_obs_steps, 3, H, W)
    'force': np.array([...])      # Shape: (n_obs_steps, 6)
}

batch = {
    'obs': obs_dict,
    'action': actions  # Shape: (horizon, 7)
}
```

### Step 2: 归一化

```python
# 在policy.compute_loss中
nobs = self.normalizer.normalize(batch['obs'])
# nobs['force'] 现在被归一化到合适的范围
```

### Step 3: 特征提取

```python
# 在DiffusionUnetHybridImagePolicy.compute_loss中

# 分离force数据和RGB数据
force_data = None
nobs_without_force = {}
for key, value in nobs.items():
    if key == 'force':
        force_data = value  # Shape: (B, n_obs_steps, 6)
    else:
        nobs_without_force[key] = value

# 编码RGB图像
rgb_features = self.obs_encoder(nobs_without_force)
# rgb_features shape: (B, n_obs_steps*rgb_feature_dim)

# 编码force数据
if self.force_encoder is not None and force_data is not None:
    # 展平时间维度: (B, n_obs_steps, 6) -> (B*n_obs_steps, 6)
    force_data_flat = force_data[:,:self.n_obs_steps,...].reshape(-1, force_data.shape[-1])
    
    # 通过MLP编码: (B*n_obs_steps, 6) -> (B*n_obs_steps, force_feature_dim)
    force_features = self.force_encoder(force_data_flat)
    
    # 恢复batch维度: (B*n_obs_steps, force_feature_dim) -> (B, n_obs_steps*force_feature_dim)
    force_features = force_features.reshape(batch_size, -1)
```

### Step 4: 特征融合

```python
# 拼接RGB特征和force特征
global_cond = torch.cat([rgb_features, force_features], dim=-1)
# global_cond shape: (B, n_obs_steps*(rgb_feature_dim + force_feature_dim))

# 输入到扩散模型
pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
```

## 4. 推理时的数据流（真机部署）

### Step 1: 从传感器读取数据

需要在`RealEnv`类中添加force传感器读取接口（需要根据实际硬件实现）：

```python
# 伪代码示例
class RealEnv:
    def __init__(self, ...):
        self.force_sensor = ForceTorqueSensor()  # 您的力传感器接口
    
    def get_obs(self):
        obs = {
            'camera_0': self.realsense.get_images(),  # 现有的相机数据
            'force': self.force_sensor.read(),         # 新增: 读取力数据
            'timestamp': time.time()
        }
        return obs
```

### Step 2: 数据预处理 (`real_inference_util.py`)

```python
# 在get_real_obs_dict中
def get_real_obs_dict(env_obs, shape_meta):
    obs_dict_np = {}
    
    for key, attr in shape_meta['obs'].items():
        if attr.get('type') == 'low_dim' and 'force' in key:
            # 直接使用force数据，保持6维
            # env_obs[key] shape: (n_obs_steps, 6)
            obs_dict_np[key] = env_obs[key]
    
    return obs_dict_np
```

### Step 3: 策略推理 (`eval_real_robot.py`)

```python
# 在策略控制循环中
obs = env.get_obs()  # 包含force数据

obs_dict_np = get_real_obs_dict(
    env_obs=obs, 
    shape_meta=cfg.task.shape_meta
)
# obs_dict_np['force'] shape: (n_obs_steps, 6)

# 转换为tensor
obs_dict = dict_apply(obs_dict_np, 
    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
# obs_dict['force'] shape: (1, n_obs_steps, 6)

# 预测动作
result = policy.predict_action(obs_dict)
action = result['action'][0]  # Shape: (n_action_steps, 7)
```

## 5. Force编码器详解

### 架构 (`force_encoder.py`)

```python
class ForceEncoder(nn.Module):
    def __init__(self, input_dim=6, output_dim=256, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 128),      # 第一层: 6 -> 128
            nn.SiLU(),              # Swish激活函数
            nn.Linear(128, 256)     # 第二层: 128 -> 256
        )
    
    def forward(self, force_data):
        # 输入: (B, 6) 或 (B, T, 6)
        # 输出: (B, 256) 或 (B, T, 256)
        return self.network(force_data)
```

### 为什么输出维度与RGB特征一致？

```python
# 在DiffusionUnetHybridImagePolicy.__init__中
rgb_feature_dim = obs_encoder.output_shape()[0]  # 例如: 64

force_encoder = ForceEncoder(
    input_dim=6,
    output_dim=rgb_feature_dim,  # 匹配RGB特征维度: 64
    hidden_dim=128
)

# 这样拼接后的特征更加平衡
total_feature_dim = rgb_feature_dim + force_feature_dim
# 例如: 64 + 64 = 128
```

## 6. 完整的数据流图

```
训练阶段:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Zarr数据集
  ├─ camera_1: (T, H, W, 3)
  ├─ force: (T, 6)          ← 力数据存储
  └─ action: (T, 7)
       ↓
Dataset加载器
  obs_dict = {
    'camera_1': (B, n_obs, 3, H, W)
    'force': (B, n_obs, 6)  ← 力数据采样
  }
       ↓
Normalizer归一化
  nobs['force']: (B, n_obs, 6)  ← 归一化后
       ↓
特征编码
  ├─ RGB Encoder → rgb_features: (B, n_obs*64)
  └─ Force Encoder → force_features: (B, n_obs*64)  ← MLP编码
       ↓
特征拼接
  global_cond = [rgb_features | force_features]
  shape: (B, n_obs*128)
       ↓
Conditional UNet 1D
  input: noisy_actions (B, horizon, 7)
  condition: global_cond
       ↓
输出: 去噪后的动作 (B, horizon, 7)


推理阶段（真机）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

力传感器硬件
  ↓ 读取实时数据
Force Sensor API
  force_reading: [fx, fy, fz, mx, my, mz]
       ↓
RealEnv.get_obs()
  obs = {
    'camera_1': [...],
    'force': np.array([...])  ← 实时力数据
  }
       ↓
real_inference_util处理
  obs_dict['force']: (n_obs_steps, 6)
       ↓
Policy.predict_action()
  [与训练阶段相同的编码和推理流程]
       ↓
输出: action (n_action_steps, 7)
  执行到机器人
```

## 7. 实际集成示例

### 需要实现的接口

```python
# 在fadp/real_world/real_env.py中需要添加

class RealEnv:
    def __init__(self, force_sensor_config=None, ...):
        # 初始化现有组件
        self.realsense = ...
        self.robot = ...
        
        # 新增: 初始化力传感器
        if force_sensor_config is not None:
            self.force_sensor = self._init_force_sensor(force_sensor_config)
        else:
            self.force_sensor = None
    
    def _init_force_sensor(self, config):
        # 根据您的硬件实现
        # 例如: ATI力传感器, Robotiq FT传感器等
        pass
    
    def get_obs(self):
        # 获取时间戳
        timestamp = time.time()
        
        # 获取相机图像
        images = self.realsense.get_images()
        
        # 获取力数据
        if self.force_sensor is not None:
            force_data = self.force_sensor.read()  # 返回 [fx,fy,fz,mx,my,mz]
        else:
            force_data = np.zeros(6)  # 如果没有传感器，返回零
        
        # 构建observation字典
        obs = {
            'timestamp': np.array([timestamp]),
            'camera_0': images['camera_0'],
            'force': force_data  # 新增
        }
        
        # 维护历史缓冲区
        self._update_obs_buffer(obs)
        
        # 返回包含n_obs_steps的历史观测
        return self._get_obs_buffer()
```

## 8. 数据采集建议

### 采集脚本修改

在数据采集时，需要同时记录force数据：

```python
# 在数据采集循环中
while collecting:
    # 读取传感器
    rgb_frame = camera.read()
    force_reading = force_sensor.read()  # [fx, fy, fz, mx, my, mz]
    
    # 存储
    episode_data['camera_0'].append(rgb_frame)
    episode_data['force'].append(force_reading)  # 新增
    episode_data['action'].append(current_action)
```

### 数据验证

采集完成后，验证force数据：

```python
import zarr
store = zarr.open('data/your_dataset/data.zarr', mode='r')

print("Force data shape:", store['force'].shape)  # 应该是 (T, 6)
print("Force data range:")
print("  fx:", store['force'][:, 0].min(), "-", store['force'][:, 0].max())
print("  fy:", store['force'][:, 1].min(), "-", store['force'][:, 1].max())
# ... 检查其他维度
```

## 9. 调试技巧

### 打印force数据流

```python
# 在训练中添加调试信息
def compute_loss(self, batch):
    nobs = self.normalizer.normalize(batch['obs'])
    
    if 'force' in nobs:
        print(f"Force data stats:")
        print(f"  Shape: {nobs['force'].shape}")
        print(f"  Mean: {nobs['force'].mean(dim=(0,1))}")
        print(f"  Std: {nobs['force'].std(dim=(0,1))}")
```

### 可视化force数据

```python
import matplotlib.pyplot as plt

# 加载数据
dataset = RealPushTImageDataset(...)
sample = dataset[0]

force_data = sample['obs']['force'].numpy()  # (n_obs_steps, 6)

# 绘制
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
labels = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']

for i, (ax, label) in enumerate(zip(axes.flat, labels)):
    ax.plot(force_data[:, i])
    ax.set_title(label)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')

plt.tight_layout()
plt.savefig('force_data_visualization.png')
```

## 总结

Force数据的关键点：

1. **配置**: 在shape_meta中定义`force: {shape: [6], type: low_dim}`
2. **存储**: 作为独立的zarr数组存储在数据集中
3. **加载**: 自动识别并加载为obs的一部分
4. **编码**: 通过ForceEncoder (MLP)编码为与RGB特征相同维度
5. **融合**: 与RGB特征拼接后作为全局条件输入扩散模型
6. **实时**: 从force传感器实时读取并处理

整个流程是**端到端**的，force数据与图像数据并行处理，最终融合为统一的条件输入。

