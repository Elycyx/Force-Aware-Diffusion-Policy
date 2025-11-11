"""
FADP Encoder: Fusion of DINOv3 visual features and Force sensor features

Architecture:
1. DINOv3 ViT-B/16 processes RGB images -> CLS token
2. MLP processes force sensor data -> force embedding
3. Bidirectional Cross Attention fuses CLS token and force embedding
4. Concatenate outputs as final feature representation

Reference: https://github.com/facebookresearch/dinov3
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from typing import Dict

try:
    from transformers import AutoImageProcessor, AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("transformers not available. Please install: pip install transformers")

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class ForceMLP(nn.Module):
    """MLP encoder for force sensor data"""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 256,
        output_dim: int = 768,  # Match DINOv2 ViT-B/16 feature dim
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Final layer to output_dim
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D) force sensor data
        Returns:
            (B, output_dim) force embedding
        """
        B, T, D = x.shape
        # Flatten temporal dimension
        x = x.reshape(B, T * D)
        return self.mlp(x)


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Cross Attention between visual CLS token and force embedding
    
    Performs:
    1. CLS -> Force: CLS token attends to force token
    2. Force -> CLS: Force token attends to CLS token
    """
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Cross attention: CLS attends to Force
        self.cls_to_force_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cls_to_force_norm = nn.LayerNorm(feature_dim)
        
        # Cross attention: Force attends to CLS
        self.force_to_cls_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.force_to_cls_norm = nn.LayerNorm(feature_dim)
        
        # Feed-forward networks
        self.cls_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        self.cls_ffn_norm = nn.LayerNorm(feature_dim)
        
        self.force_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        self.force_ffn_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, cls_token, force_token):
        """
        Args:
            cls_token: (B, feature_dim) visual CLS token
            force_token: (B, feature_dim) force embedding
        Returns:
            cls_enhanced: (B, feature_dim) enhanced CLS token
            force_enhanced: (B, feature_dim) enhanced force token
        """
        # Add sequence dimension for attention: (B, 1, D)
        cls_token = cls_token.unsqueeze(1)
        force_token = force_token.unsqueeze(1)
        
        # Cross attention: CLS attends to Force
        cls_attn_out, _ = self.cls_to_force_attn(
            query=cls_token,
            key=force_token,
            value=force_token
        )
        cls_token = self.cls_to_force_norm(cls_token + cls_attn_out)
        
        # Feed-forward for CLS
        cls_ffn_out = self.cls_ffn(cls_token)
        cls_token = self.cls_ffn_norm(cls_token + cls_ffn_out)
        
        # Cross attention: Force attends to CLS
        force_attn_out, _ = self.force_to_cls_attn(
            query=force_token,
            key=cls_token,
            value=cls_token
        )
        force_token = self.force_to_cls_norm(force_token + force_attn_out)
        
        # Feed-forward for Force
        force_ffn_out = self.force_ffn(force_token)
        force_token = self.force_ffn_norm(force_token + force_ffn_out)
        
        # Remove sequence dimension: (B, 1, D) -> (B, D)
        cls_enhanced = cls_token.squeeze(1)
        force_enhanced = force_token.squeeze(1)
        
        return cls_enhanced, force_enhanced


class FADPEncoder(ModuleAttrMixin):
    """
    FADP Encoder combining DINOv3 visual features and force sensor features
    
    Features:
    - DINOv3 ViT-B/16 for RGB images
    - MLP encoder for force sensor data
    - Bidirectional cross attention for feature fusion
    - Outputs concatenated [cls_token, force_token]
    """
    
    def __init__(
        self,
        shape_meta: dict,
        # DINOv3 settings
        dinov3_model: str = 'facebook/dinov3-vit-base-pretrain-lvd1689m',  # HF model name
        dinov3_frozen: bool = False,
        use_huggingface: bool = True,  # Use HuggingFace by default
        # Force encoder settings
        force_mlp_hidden_dim: int = 256,
        force_mlp_layers: int = 3,
        # Cross attention settings
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        # Transform settings
        transforms: list = None,
    ):
        """
        Args:
            shape_meta: Dictionary containing observation shapes
            dinov3_model: DINOv3 model name
                - HuggingFace format (default): 'facebook/dinov3-vit-base-pretrain-lvd1689m'
                - Legacy format: 'dinov3_vitb16' (requires use_huggingface=False)
            dinov3_frozen: Whether to freeze DINOv3 weights
            use_huggingface: Whether to use HuggingFace transformers (recommended)
            force_mlp_hidden_dim: Hidden dimension for force MLP
            force_mlp_layers: Number of layers in force MLP
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout rate for attention
            transforms: Image transforms (e.g., RandomCrop, Resize)
        """
        super().__init__()
        
        # Parse observation shapes
        rgb_keys = []
        force_keys = []
        key_shape_map = {}
        
        obs_shape_meta = shape_meta['obs']
        image_shape = None
        force_dim = 0
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type_name = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            
            if type_name == 'rgb':
                rgb_keys.append(key)
                if image_shape is None:
                    image_shape = shape[1:]  # (C, H, W)
                else:
                    assert image_shape == shape[1:], "All RGB inputs must have same shape"
            
            elif type_name == 'low_dim' and not attr.get('ignore_by_policy', False):
                # Check if this is force data (key is 'force' based on convert_session_to_zarr.py)
                if key == 'force' or 'force' in key.lower():
                    force_keys.append(key)
                    force_dim += shape[0]
        
        if len(rgb_keys) == 0:
            raise ValueError("No RGB observations found in shape_meta")
        
        # Get horizon info for logging
        rgb_horizon = obs_shape_meta[rgb_keys[0]]['horizon'] if rgb_keys else 0
        force_horizon = obs_shape_meta[force_keys[0]]['horizon'] if force_keys else 0
        
        logger.info(f"RGB keys: {rgb_keys}, horizon: {rgb_horizon}")
        logger.info(f"Force keys: {force_keys}, horizon: {force_horizon}")
        logger.info(f"Force dimension: {force_dim}")
        
        # Load DINOv3 model
        # Available HuggingFace models:
        # ViT models: 
        #   facebook/dinov3-vit-small-pretrain-lvd1689m
        #   facebook/dinov3-vit-base-pretrain-lvd1689m (default)
        #   facebook/dinov3-vit-large-pretrain-lvd1689m
        #   facebook/dinov3-vit-giant-pretrain-lvd1689m
        # ConvNeXt models:
        #   facebook/dinov3-convnext-tiny-pretrain-lvd1689m
        #   facebook/dinov3-convnext-small-pretrain-lvd1689m
        #   facebook/dinov3-convnext-base-pretrain-lvd1689m
        #   facebook/dinov3-convnext-large-pretrain-lvd1689m
        
        self.use_huggingface = use_huggingface
        
        if use_huggingface:
            # Load from HuggingFace (recommended)
            if not HF_AVAILABLE:
                raise RuntimeError(
                    "transformers is not installed. Please install it:\n"
                    "pip install transformers"
                )
            
            logger.info(f"Loading DINOv3 from HuggingFace: {dinov3_model}")
            try:
                # Load image processor
                self.image_processor = AutoImageProcessor.from_pretrained(dinov3_model)
                
                # Load model
                dinov3 = AutoModel.from_pretrained(
                    dinov3_model,
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded {dinov3_model} from HuggingFace")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load DINOv3 from HuggingFace: {e}\n"
                    f"Model: {dinov3_model}\n"
                    f"Please ensure:\n"
                    f"1. Model name is correct (e.g., facebook/dinov3-vit-base-pretrain-lvd1689m)\n"
                    f"2. transformers is installed: pip install transformers\n"
                    f"3. You have internet connection or model is cached"
                )
        else:
            # Legacy: Load from torch hub
            logger.warning("Using legacy torch hub loading. Consider using HuggingFace instead.")
            try:
                dinov3 = torch.hub.load('facebookresearch/dinov3', dinov3_model)
                logger.info(f"Successfully loaded {dinov3_model} from torch hub")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load DINOv3 from torch hub: {e}\n"
                    f"Consider using HuggingFace instead:\n"
                    f"dinov3_model='facebook/dinov3-vit-base-pretrain-lvd1689m'\n"
                    f"use_huggingface=True"
                )
        
        # Get feature dimension from DINOv3
        if use_huggingface:
            # For HuggingFace models, use config
            if hasattr(dinov3, 'config'):
                if hasattr(dinov3.config, 'hidden_size'):
                    feature_dim = dinov3.config.hidden_size
                else:
                    feature_dim = 768  # Default
            else:
                feature_dim = 768
        else:
            # Legacy: infer from model attributes
            if hasattr(dinov3, 'embed_dim'):
                feature_dim = dinov3.embed_dim
            elif hasattr(dinov3, 'num_features'):
                feature_dim = dinov3.num_features
            else:
                # Infer from model name
                if 'small' in dinov3_model.lower():
                    feature_dim = 384
                elif 'base' in dinov3_model.lower():
                    feature_dim = 768
                elif 'large' in dinov3_model.lower():
                    feature_dim = 1024
                elif 'giant' in dinov3_model.lower():
                    feature_dim = 1536
                elif 'convnext' in dinov3_model.lower():
                    if 'tiny' in dinov3_model.lower():
                        feature_dim = 384
                    elif 'small' in dinov3_model.lower():
                        feature_dim = 768
                    elif 'base' in dinov3_model.lower():
                        feature_dim = 1024
                    elif 'large' in dinov3_model.lower():
                        feature_dim = 1536
                    else:
                        feature_dim = 768
                else:
                    feature_dim = 768
        
        logger.info(f"DINOv3 feature dimension: {feature_dim}")
        
        # Freeze DINOv3 if requested
        if dinov3_frozen:
            for param in dinov3.parameters():
                param.requires_grad = False
            logger.info("DINOv3 weights frozen")
        
        # Setup image transforms (only for non-HuggingFace mode)
        if not use_huggingface:
            if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
                # Convert config-based transforms to torch transforms
                assert transforms[0].type == 'RandomCrop'
                ratio = transforms[0].ratio
                transforms_list = [
                    torchvision.transforms.RandomCrop(size=int(image_shape[1] * ratio)),
                    torchvision.transforms.Resize(size=image_shape[1], antialias=True)
                ] + transforms[1:]
                self.rgb_transform = nn.Sequential(*transforms_list)
            else:
                self.rgb_transform = nn.Identity() if transforms is None else nn.Sequential(*transforms)
        else:
            # HuggingFace uses its own image processor
            self.rgb_transform = None
            logger.info("Using HuggingFace image processor, custom transforms will be ignored")
        
        # Create Force MLP encoder (if force data exists)
        self.has_force = force_dim > 0
        if self.has_force:
            # Calculate input dimension: force_horizon * force_dim
            force_horizon = obs_shape_meta[force_keys[0]]['horizon']
            force_input_dim = force_horizon * force_dim
            
            logger.info(f"Force MLP input dimension: {force_input_dim} = {force_horizon} (horizon) × {force_dim} (force_dim)")
            
            self.force_encoder = ForceMLP(
                input_dim=force_input_dim,
                hidden_dim=force_mlp_hidden_dim,
                output_dim=feature_dim,
                num_layers=force_mlp_layers,
                dropout=attention_dropout
            )
            
            # Create bidirectional cross attention
            self.cross_attention = BidirectionalCrossAttention(
                feature_dim=feature_dim,
                num_heads=num_attention_heads,
                dropout=attention_dropout
            )
            
            output_feature_dim = feature_dim * 2  # Concatenate CLS and Force tokens
        else:
            logger.warning("No force data found, using only visual features")
            self.force_encoder = None
            self.cross_attention = None
            output_feature_dim = feature_dim
        
        # Store attributes
        self.dinov3 = dinov3
        self.shape_meta = shape_meta
        self.rgb_keys = sorted(rgb_keys)
        self.force_keys = sorted(force_keys)
        self.key_shape_map = key_shape_map
        self.feature_dim = feature_dim
        self.output_feature_dim = output_feature_dim
        self.force_dim = force_dim
        
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            obs_dict: Dictionary with keys:
                - RGB keys: (B, T, C, H, W)
                - Force keys: (B, T, force_dim)
        
        Returns:
            features: (B, output_feature_dim) concatenated features
        """
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # Process RGB images with DINOv3
        cls_tokens = []
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            
            # Reshape to (B*T, C, H, W)
            img = img.reshape(B * T, *img.shape[2:])
            
            if self.use_huggingface:
                # HuggingFace preprocessing
                # Input: (B*T, C, H, W) in range [0, 1]
                # Need to convert to format expected by image processor
                
                # Convert tensor to numpy for image processor
                # Shape: (B*T, C, H, W) -> (B*T, H, W, C)
                img_numpy = img.permute(0, 2, 3, 1).cpu().numpy()
                
                # Convert to uint8 [0, 255] range as expected by most processors
                img_numpy = (img_numpy * 255).astype('uint8')
                
                # Process with HuggingFace image processor
                # It will handle resizing, normalization, etc.
                inputs = self.image_processor(images=list(img_numpy), return_tensors="pt")
                
                # Move to same device as model
                inputs = {k: v.to(self.dinov3.device) for k, v in inputs.items()}
                
                # Forward through DINOv3
                with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.dinov3.parameters())):
                    outputs = self.dinov3(**inputs)
                
                # Get pooled output (CLS token equivalent)
                # HuggingFace models return BaseModelOutput with pooler_output
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    cls_token = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # Use [CLS] token (first token) from last_hidden_state
                    cls_token = outputs.last_hidden_state[:, 0]
                else:
                    raise RuntimeError("Unexpected output format from HuggingFace model")
            else:
                # Legacy: apply transforms and forward
                img = self.rgb_transform(img)
                
                with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.dinov3.parameters())):
                    dinov3_output = self.dinov3(img)
                
                # Get CLS token
                if isinstance(dinov3_output, dict):
                    cls_token = dinov3_output['x_norm_clstoken']
                else:
                    cls_token = dinov3_output
            
            # Reshape to (B, T*feature_dim)
            cls_token = cls_token.reshape(B, T * self.feature_dim)
            cls_tokens.append(cls_token)
        
        # Average CLS tokens from all RGB inputs
        visual_feature = torch.stack(cls_tokens, dim=0).mean(dim=0)  # (B, T*feature_dim)
        
        # For simplicity, use only the first timestep's feature
        # or average across time
        visual_feature = visual_feature.reshape(B, -1, self.feature_dim).mean(dim=1)  # (B, feature_dim)
        
        if not self.has_force:
            # Return only visual features
            return visual_feature
        
        # Process force data
        force_data_list = []
        for key in self.force_keys:
            force_data = obs_dict[key]
            B, T = force_data.shape[:2]
            assert B == batch_size
            force_data_list.append(force_data)
        
        # Concatenate all force data along feature dimension
        force_data = torch.cat(force_data_list, dim=-1)  # (B, T, force_dim)
        
        # Encode force data
        force_feature = self.force_encoder(force_data)  # (B, feature_dim)
        
        # Apply bidirectional cross attention
        cls_enhanced, force_enhanced = self.cross_attention(visual_feature, force_feature)
        
        # Concatenate features
        output_feature = torch.cat([cls_enhanced, force_enhanced], dim=-1)  # (B, 2*feature_dim)
        
        return output_feature
    
    @torch.no_grad()
    def output_shape(self):
        """
        Compute output shape
        
        Returns:
            shape: (1, output_feature_dim)
        """
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta['obs']
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            horizon = attr['horizon']
            this_obs = torch.zeros(
                (1, horizon) + shape,
                dtype=self.dtype,
                device=self.device
            )
            example_obs_dict[key] = this_obs
        
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__ == '__main__':
    # Test the encoder
    # Note: 'force' key matches the actual dataset key from convert_session_to_zarr.py
    shape_meta = {
        'obs': {
            'camera0_rgb': {
                'shape': [3, 224, 224],
                'horizon': 2,
                'type': 'rgb'
            },
            'force': {
                'shape': [6],  # [fx, fy, fz, mx, my, mz]
                'horizon': 2,
                'type': 'low_dim'
            }
        }
    }
    
    encoder = FADPEncoder(
        shape_meta=shape_meta,
        dinov3_model='dinov3_vitb16',
        dinov3_frozen=False
    )
    
    obs_dict = {
        'camera0_rgb': torch.randn(2, 2, 3, 224, 224),
        'force': torch.randn(2, 2, 6)
    }
    
    output = encoder(obs_dict)
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, {encoder.output_feature_dim})")

