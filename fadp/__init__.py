"""
Force-Aware Diffusion Policy (FADP)

A robot manipulation policy that integrates force/torque sensor data
with visual observations using diffusion models.
"""

__version__ = '1.0.0'
__author__ = 'FADP Contributors'
__license__ = 'MIT'

from fadp.policy.force_aware_diffusion_policy import ForceAwareDiffusionPolicy
from fadp.model.common.force_encoder import ForceEncoder
from fadp.dataset.fadp_dataset import FADPDataset
from fadp.dataset.real_pusht_image_dataset import RealPushTImageDataset

__all__ = [
    '__version__',
    'ForceAwareDiffusionPolicy',
    'ForceEncoder',
    'FADPDataset',
    'RealPushTImageDataset',
]

