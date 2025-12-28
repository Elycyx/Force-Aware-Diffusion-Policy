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
import sys
import os
from typing import Dict

# Add model directory to sys.path to allow imports from depth_anything_3
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, '..'))
if model_dir not in sys.path:
    sys.path.append(model_dir)

try:
    from transformers import AutoImageProcessor, AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("transformers not available. Please install: pip install transformers")

try:
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.utils.model_loading import load_pretrained_weights
    DA3_AVAILABLE = True
except ImportError:
    DA3_AVAILABLE = False
    logging.warning("depth_anything_3 not available in path.")

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class ForceMLP(nn.Module):
    """
    MLP encoder for force sensor data
    
    Architecture optimizations:
    1. GELU activation: Force/torque signals contain positive and negative values.
       ReLU has zero gradient for negative inputs, causing "dead neurons" and poor
       fitting for negative changes. GELU is smooth, efficient, and performs well
       in modern encoders (like Transformers).
       
    2. Post-Normalization: LayerNorm is applied after activation and dropout.
       This ordering (Post-Normalization) is proven to be more stable and 
       converges faster in deep networks and Transformer architectures.
       
       Original: Linear -> LayerNorm -> ReLU -> Dropout
       Optimized: Linear -> GELU -> Dropout -> LayerNorm
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 256,
        output_dim: int = 768,  # Match DINOv3 ViT-B/16 feature dim
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            # Optimized ordering: Linear -> GELU -> Dropout -> LayerNorm
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),  # GELU instead of ReLU for better handling of negative values
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)  # Post-normalization for better stability
            ])
            current_dim = hidden_dim
        
        # Final layer to output_dim (no activation, just linear projection + normalization)
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
        # Depth Anything 3 settings
        use_depth: bool = False,
        depth_model_name: str = 'da3-small',
        depth_checkpoint_path: str = None,
        depth_frozen: bool = True,
        # Velocity settings
        use_velocity: bool = False,
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
            use_depth: Whether to use Depth Anything 3 encoder
            depth_model_name: Depth Anything 3 model name (e.g. 'da3-small')
            depth_checkpoint_path: Path to Depth Anything 3 checkpoint
            depth_frozen: Whether to freeze Depth Anything 3 weights
            use_velocity: Whether to append velocity estimate (7D) to encoder output
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
        self.use_velocity = use_velocity
        self.velocity_dim = 7 if use_velocity else 0
        # avoid spamming logs
        self._warned_missing_velocity = False
        
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
                # Exclude velocity: it is appended directly without network processing
                if key == 'velocity' or 'velocity' in key.lower():
                    continue
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

        # Initialize Depth Anything 3 if requested
        self.use_depth = use_depth
        self.depth_feature_dim = 0
        if use_depth:
            if not DA3_AVAILABLE:
                raise RuntimeError("depth_anything_3 not available. Cannot use depth encoder.")
            
            logger.info(f"Loading Depth Anything 3: {depth_model_name}")
            
            # Try to detect if depth_checkpoint_path is a HuggingFace Hub ID
            # (contains '/' and is not a local path)
            use_hf_hub = False
            if depth_checkpoint_path is not None:
                # Check if it looks like a HuggingFace Hub ID (e.g., "depth-anything/DA3-SMALL")
                if '/' in depth_checkpoint_path and not os.path.exists(depth_checkpoint_path):
                    use_hf_hub = True
            
            if use_hf_hub:
                # Use HuggingFace Hub API to load model (auto-download)
                logger.info(f"Loading DA3 from HuggingFace Hub: {depth_checkpoint_path}")
                try:
                    from depth_anything_3.api import DepthAnything3
                    # Load complete model from HuggingFace
                    full_model = DepthAnything3.from_pretrained(depth_checkpoint_path)
                    # Extract only the backbone (we don't need inference pipeline)
                    self.depth_model = full_model.model.backbone
                    logger.info(f"Successfully loaded DA3 from HuggingFace Hub")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load DA3 from HuggingFace Hub: {e}\n"
                        f"Please ensure:\n"
                        f"1. The model ID is correct (e.g., 'depth-anything/DA3-SMALL')\n"
                        f"2. You have internet connection or the model is cached\n"
                        f"3. huggingface_hub is installed: pip install huggingface_hub"
                    )
            else:
                # Load from local config + optional checkpoint
                da3_config_path = os.path.join(model_dir, 'depth_anything_3', 'configs', f'{depth_model_name}.yaml')
                if not os.path.exists(da3_config_path):
                    raise FileNotFoundError(f"DA3 config not found at {da3_config_path}")
                
                da3_cfg = load_config(da3_config_path)
                full_model = create_object(da3_cfg)
                
                if depth_checkpoint_path is not None and os.path.exists(depth_checkpoint_path):
                    logger.info(f"Loading DA3 weights from local file: {depth_checkpoint_path}")
                    load_pretrained_weights(full_model, depth_checkpoint_path)
                
                # Extract backbone
                self.depth_model = full_model.backbone
            
            # Freeze DA3 if requested
            if depth_frozen:
                for param in self.depth_model.parameters():
                    param.requires_grad = False
                logger.info("DA3 weights frozen")
                
            # Determine depth feature dimension
            # Based on observation and DA3 implementation, the backbone outputs features 
            # with dimension 2 * embed_dim (concatenation of features)
            if hasattr(self.depth_model, 'embed_dim'):
                self.depth_feature_dim = self.depth_model.embed_dim * 2
            else:
                # Fallback based on model name
                if 'small' in depth_model_name:
                    self.depth_feature_dim = 384 * 2
                elif 'base' in depth_model_name:
                    self.depth_feature_dim = 768 * 2
                elif 'large' in depth_model_name:
                    self.depth_feature_dim = 1024 * 2
                elif 'giant' in depth_model_name:
                    self.depth_feature_dim = 1536 * 2
                else:
                    self.depth_feature_dim = 384 * 2 # Default assumption
            
            logger.info(f"DA3 feature dimension: {self.depth_feature_dim} (2x embed_dim)")
            
            # CRITICAL: Force clear PositionGetter cache
            # Even without dummy forward, the pre-trained model loading or internal initialization
            # might have cached CPU tensors. We must clear it to ensure GPU tensors are generated during training.
            model_to_check = self.depth_model
            if hasattr(model_to_check, 'pretrained'):
                model_to_check = model_to_check.pretrained
                
            if hasattr(model_to_check, 'position_getter') and hasattr(model_to_check.position_getter, 'position_cache'):
                model_to_check.position_getter.position_cache.clear()
                logger.info("Forcibly cleared PositionGetter cache to prevent device mismatch")
            
            # ImageNet mean/std for normalization
            self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Add depth feature dim if used
        if self.use_depth:
             # Calculate combined feature dimension
             # ForceMLP and CrossAttention will operate in this dimension
             feature_dim += self.depth_feature_dim

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
        
        # NOTE: self.feature_dim tracks the "main" processing dimension
        # (used for ForceMLP output and CrossAttention dimension)
        # If depth is used, it INCLUDES depth dimension.
        # If NOT used, it is just DINOv3 dimension.
        # However, we must be careful when reshaping DINOv3 output.
        self.feature_dim = feature_dim
        
        # Store original DINOv3 feature dimension for reshaping
        if use_huggingface:
            if hasattr(dinov3, 'config') and hasattr(dinov3.config, 'hidden_size'):
                self.dinov3_feature_dim = dinov3.config.hidden_size
            else:
                self.dinov3_feature_dim = 768
        else:
             # Logic copied from above
             if hasattr(dinov3, 'embed_dim'):
                self.dinov3_feature_dim = dinov3.embed_dim
             elif hasattr(dinov3, 'num_features'):
                self.dinov3_feature_dim = dinov3.num_features
             else:
                # Infer from model name (fallback)
                if 'small' in dinov3_model.lower():
                    self.dinov3_feature_dim = 384
                elif 'base' in dinov3_model.lower():
                    self.dinov3_feature_dim = 768
                elif 'large' in dinov3_model.lower():
                    self.dinov3_feature_dim = 1024
                elif 'giant' in dinov3_model.lower():
                    self.dinov3_feature_dim = 1536
                elif 'convnext' in dinov3_model.lower():
                     if 'tiny' in dinov3_model.lower():
                        self.dinov3_feature_dim = 384
                     elif 'small' in dinov3_model.lower():
                        self.dinov3_feature_dim = 768
                     elif 'base' in dinov3_model.lower():
                        self.dinov3_feature_dim = 1024
                     elif 'large' in dinov3_model.lower():
                        self.dinov3_feature_dim = 1536
                     else:
                        self.dinov3_feature_dim = 768
                else:
                    self.dinov3_feature_dim = 768

        # If force is used, output is 2 * feature_dim (CLS + Force)
        # If no force, output is feature_dim (CLS)
        # Note: feature_dim now includes depth if use_depth=True
        if self.has_force:
            self.output_feature_dim = feature_dim * 2
        else:
            self.output_feature_dim = feature_dim
        if self.use_velocity:
            self.output_feature_dim = self.output_feature_dim + self.velocity_dim
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
        
        # Determine current device from input
        device = next(iter(obs_dict.values())).device

        # Defense against PositionGetter cache device mismatch
        if self.use_depth:
            # Check if depth_model is DINOv2 wrapper or DinoVisionTransformer directly
            model_to_check = self.depth_model
            if hasattr(model_to_check, 'pretrained'):
                model_to_check = model_to_check.pretrained
            
            if hasattr(model_to_check, 'position_getter'):
                pg = model_to_check.position_getter
                if hasattr(pg, 'position_cache'):
                    cache = pg.position_cache
                    if len(cache) > 0:
                        first_key = next(iter(cache.keys()))
                        first_val = cache[first_key]
                        
                        if first_val.device != device:
                            logger.warning(f"PositionGetter cache device mismatch detected! Cache: {first_val.device}, Input: {device}")
                            cache.clear()
                            logger.info("Cleared PositionGetter cache.")
            else:
                # pass
                logger.warning(f"Depth model {type(self.depth_model)} (or inner) does not have position_getter")

        # Process RGB images with DINOv3
        cls_tokens = []
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            
            # ... (validation code omitted) ...
            
            # Reshape to (B*T, C, H, W)
            img = img.reshape(B * T, *img.shape[2:])
            
            if self.use_huggingface:
                # ... (preprocessing omitted) ...
                
                # Convert to uint8 [0, 255] range as expected by most processors
                img_numpy = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                
                # Process with HuggingFace image processor
                inputs = self.image_processor(images=list(img_numpy), return_tensors="pt")
                inputs = {k: v.to(self.dinov3.device) for k, v in inputs.items()}
                
                with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.dinov3.parameters())):
                    outputs = self.dinov3(**inputs)
                
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    cls_token = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    cls_token = outputs.last_hidden_state[:, 0]
                else:
                    raise RuntimeError("Unexpected output format from HuggingFace model")
            else:
                # Legacy path
                img = self.rgb_transform(img)
                with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.dinov3.parameters())):
                    dinov3_output = self.dinov3(img)
                
                if isinstance(dinov3_output, dict):
                    cls_token = dinov3_output['x_norm_clstoken']
                else:
                    cls_token = dinov3_output
            
            # Reshape to (B, T*feature_dim) -> CORRECTED to use self.dinov3_feature_dim
            cls_token = cls_token.reshape(B, T * self.dinov3_feature_dim)
            cls_tokens.append(cls_token)
        
        # Average CLS tokens from all RGB inputs
        visual_feature = torch.stack(cls_tokens, dim=0).mean(dim=0)  # (B, T*dinov3_feature_dim)
        
        # For simplicity, use only the first timestep's feature
        # or average across time
        visual_feature = visual_feature.reshape(B, -1, self.dinov3_feature_dim).mean(dim=1)  # (B, dinov3_feature_dim)
        
        # Process with Depth Anything 3 if enabled
        depth_feature = None
        if self.use_depth:
            depth_tokens = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                # img is (B, T, C, H, W) in [0, 1]
                B, T, C, H, W = img.shape
                
                # Reshape to (B*T, C, H, W)
                img_flat = img.reshape(B * T, C, H, W)
                
                # Normalize with ImageNet mean/std
                img_norm = (img_flat - self.imagenet_mean) / self.imagenet_std
                
                # DA3 internally handles resizing but we want to be safe or just pass as is
                # For now pass as is, assuming inputs are suitable or DA3 handles it.
                
                # Center crop to remove black borders from fisheye images (optional)
                # Keep the largest square that fits inside the circle? 
                # Or just crop to a smaller central region.
                # Assuming input is square (H=W), fisheye circle usually touches edges.
                # To remove corners, we might need a mask or just crop a central square.
                # However, DINO/Depth models usually take square inputs.
                # If we crop, we lose FOV. 
                # Let's crop to the central region if requested (simple center crop).
                # For a circle inscribed in a square of side L, the largest square inscribed in the circle has side L/sqrt(2) ≈ 0.707*L.
                
                # We apply a center crop with ratio ~0.707 to remove most black corners while keeping the center.
                # But let's make it configurable or just hardcode a safe ratio for fisheye.
                # Given user request "仅保留圆内切的矩形区域" (keep only the rectangle inscribed in the circle)
                
                # Note: This logic only applies to Depth Anything input.
                # RGB encoder (DINOv3) receives the original image via self.rgb_transform or image_processor above.
                
                H_in, W_in = img_norm.shape[-2:]
                if H_in == W_in:
                    # Use a more aggressive crop ratio to remove black borders
                    # Original: L/sqrt(2) ≈ 0.707L (theoretical inscribed square)
                    # Use 0.6L to be more conservative and remove more black edges
                    crop_ratio = 0.5  # More aggressive than 0.707
                    crop_size = int(H_in * crop_ratio)
                    start = (H_in - crop_size) // 2
                    img_norm_cropped = img_norm[..., start:start+crop_size, start:start+crop_size]
                    
                    # Resize to 224x224 (standard input size for ViT models)
                    img_norm = F.interpolate(img_norm_cropped, size=(224, 224), mode='bilinear', align_corners=False)

                # Add sequence dimension (S=1) as expected by Depth Anything 3 / DINOv2
                # shape: (B*T, C, H, W) -> (B*T, 1, C, H, W)
                img_norm = img_norm.unsqueeze(1)

                with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.depth_model.parameters())):
                    # Get features from backbone (self.depth_model is the backbone)
                    # DA3 backbone returns list of features from different layers
                    # We want the last one for high level features
                    feats, _ = self.depth_model(img_norm)
                    
                    # feats is a list of tuples (patch_tokens, camera_tokens)
                    # Note: camera_tokens in DA3 implementation might be flawed for batch_size > 1
                    # So we use patch_tokens and perform Global Average Pooling (GAP)
                    
                    last_feat = feats[-1]
                    if isinstance(last_feat, tuple):
                        # last_feat[0] is patch_tokens: (B*T, S, N, D)
                        patch_tokens = last_feat[0]
                        
                        # Perform GAP over spatial/token dimensions
                        if patch_tokens.ndim == 4: # (B*T, S, N, D)
                            cls_token = patch_tokens.mean(dim=(1, 2)) # Average over S and N
                        elif patch_tokens.ndim == 3: # (B*T, N, D)
                            cls_token = patch_tokens.mean(dim=1) # Average over N
                        else:
                            raise RuntimeError(f"Unexpected patch_tokens shape: {patch_tokens.shape}")
                    else:
                        # Fallback: global average pooling
                        if last_feat.ndim == 4: # (B*T, S, N, D)
                             cls_token = last_feat.mean(dim=(1, 2)) 
                        elif last_feat.ndim == 3: # (B*T, N, D)
                             cls_token = last_feat.mean(dim=1)
                        else:
                             # Assume it's already pooled?
                             cls_token = last_feat
                
                # Reshape back to (B, T, D)
                cls_token = cls_token.reshape(B, T, -1)
                depth_tokens.append(cls_token)
            
            # Average across cameras and time
            depth_feature = torch.stack(depth_tokens, dim=0).mean(dim=0) # (B, T, D)
            depth_feature = depth_feature.mean(dim=1) # (B, D)
            
            # Fuse Visual and Depth features
            # Concatenate along feature dimension
            visual_feature = torch.cat([visual_feature, depth_feature], dim=-1)

        if not self.has_force:
            # Return combined visual + depth features
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

        # Append velocity (no network processing)
        if self.use_velocity:
            if 'velocity' in obs_dict:
                vel = obs_dict['velocity']
                # expected (B, T, 7) or (B, 7)
                if vel.ndim == 3:
                    vel = vel[:, -1, :]
                elif vel.ndim == 2:
                    pass
                else:
                    raise RuntimeError(f"Unexpected velocity shape: {vel.shape}")
            else:
                if not self._warned_missing_velocity:
                    logger.warning(
                        "use_velocity=True but 'velocity' not found in obs_dict. "
                        "FADPEncoder will use zeros for velocity (this will not crash, but may hurt performance). "
                        "Deployment: pass obs_dict['velocity'] with shape (B, 7) or (B, T, 7)."
                    )
                    self._warned_missing_velocity = True
                vel = torch.zeros((output_feature.shape[0], self.velocity_dim), device=output_feature.device, dtype=output_feature.dtype)
            output_feature = torch.cat([output_feature, vel], dim=-1)

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
        
        # CRITICAL: Clear PositionGetter cache after output_shape calculation
        # output_shape() is typically called on CPU during initialization.
        # This populates PositionGetter cache with CPU tensors.
        # Subsequent training on GPU will fail because PositionGetter returns cached CPU tensors.
        model_to_check = self.depth_model
        if hasattr(model_to_check, 'pretrained'):
            model_to_check = model_to_check.pretrained
            
        if hasattr(model_to_check, 'position_getter') and hasattr(model_to_check.position_getter, 'position_cache'):
            model_to_check.position_getter.position_cache.clear()
            # logger.info("Cleared PositionGetter cache after output_shape calculation")

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

