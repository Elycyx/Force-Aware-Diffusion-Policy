"""
Force-Aware Diffusion Policy (FADP)

A comprehensive framework for robot manipulation learning using diffusion policies
with force/torque integration.
"""

__version__ = '1.0.0'

# Make key modules easily accessible
from diffusion_policy.dataset import fadp_dataset, umi_dataset
from diffusion_policy.policy import diffusion_unet_fadp_policy, diffusion_unet_timm_policy
from diffusion_policy.model.vision import fadp_encoder, timm_obs_encoder

__all__ = [
    'fadp_dataset',
    'umi_dataset',
    'diffusion_unet_fadp_policy',
    'diffusion_unet_timm_policy',
    'fadp_encoder',
    'timm_obs_encoder',
]

