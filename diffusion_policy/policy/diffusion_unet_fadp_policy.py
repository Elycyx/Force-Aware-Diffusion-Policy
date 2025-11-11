from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.fadp_encoder import FADPEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionUnetFADPPolicy(BaseImagePolicy):
    """
    Diffusion Policy with FADP Encoder (DINOv3 + Force MLP + Cross Attention)
    
    This policy uses:
    - FADPEncoder for processing RGB images and force sensor data
    - Conditional UNet1D for diffusion-based action generation
    """
    
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: FADPEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            force_loss_weight=1.0,  # 力预测loss的权重
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())
        
        # 判断action维度: 7维(仅action) 或 13维(action+force)
        self.has_force_prediction = (action_dim == 13)
        if self.has_force_prediction:
            self.action_only_dim = 7
            self.force_dim = 6
        else:
            self.action_only_dim = action_dim
            self.force_dim = 0


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.force_loss_weight = force_loss_weight
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        预测动作和力
        
        Args:
            obs_dict: 观测字典，必须包含 "obs" key
            fixed_action_prefix: 未归一化的action前缀（用于inpainting）
            
        Returns:
            result: 字典包含:
                - 'action_pred': 完整预测 (B, T, action_dim)
                  - 如果 action_dim=7: [x, y, z, rx, ry, rz, gripper]
                  - 如果 action_dim=13: [x, y, z, rx, ry, rz, gripper, fx, fy, fz, mx, my, mz]
                - 'action': 与 'action_pred' 相同（向后兼容）
                - 'action_only': 仅action部分 (B, T, 7) - 仅当 action_dim=13 时存在
                - 'force_pred': 预测的力 (B, T, 6) - 仅当 action_dim=13 时存在
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        # 构建结果字典
        result = {
            'action': action_pred, # (B, T, action_dim)
            'action_pred': action_pred
        }
        
        # 如果包含力预测，分离action和force
        if self.has_force_prediction:
            result['action'] = action_pred[..., :7]
            force_delta = action_pred[..., 7:]   # (B, T, 6) - 相对力（增量）
            
            # 将相对力转换为绝对力
            # 相对力是相对于当前观测的最后一个force的增量
            if 'force' in obs_dict:
                # 获取当前观测的最后一个force值 (B, T_obs, 6)
                current_force = obs_dict['force'][:, -1:, :]  # (B, 1, 6) - 取最后一个时间步
                # 广播加法: (B, T, 6) + (B, 1, 6) -> (B, T, 6)
                force_abs = force_delta + current_force
                result['force_pred'] = force_abs  # 绝对力
                result['force_delta'] = force_delta  # 相对力（增量）
            else:
                # 如果没有观测force，只返回相对力
                result['force_pred'] = force_delta  # 此时force_pred实际上是相对力
                result['force_delta'] = force_delta
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, return_components=False):
        """
        计算训练loss
        
        Args:
            batch: 包含obs和action的batch数据
            return_components: 是否返回loss的各个分量（用于logging）
            
        Returns:
            如果return_components=False: 返回标量loss（向后兼容）
            如果return_components=True: 返回字典，包含：
                - 'loss': 总loss
                - 'loss_action': action部分的loss（如果有force预测）
                - 'loss_force': force部分的loss（如果有force预测）
        """
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 分别计算action和force的loss
        if self.has_force_prediction:
            # 13维: 前7维是action，后6维是force
            # 分离预测和真实的action与force部分
            pred_action = pred[..., :7]  # (B, T, 7)
            pred_force = pred[..., 7:]   # (B, T, 6)
            
            target_action = target[..., :7]
            target_force = target[..., 7:]
            
            # 分别计算loss
            loss_action = F.mse_loss(pred_action, target_action, reduction='none')
            loss_action = loss_action.type(loss_action.dtype)
            loss_action = reduce(loss_action, 'b ... -> b (...)', 'mean')
            loss_action = loss_action.mean()
            
            loss_force = F.mse_loss(pred_force, target_force, reduction='none')
            loss_force = loss_force.type(loss_force.dtype)
            loss_force = reduce(loss_force, 'b ... -> b (...)', 'mean')
            loss_force = loss_force.mean()
            
            # 加权合并
            loss = loss_action + self.force_loss_weight * loss_force
            
            # 返回loss分量
            if return_components:
                return {
                    'loss': loss,
                    'loss_action': loss_action,
                    'loss_force': loss_force,
                    'loss_force_weighted': self.force_loss_weight * loss_force
                }
        else:
            # 7维: 仅action
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss.type(loss.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)

