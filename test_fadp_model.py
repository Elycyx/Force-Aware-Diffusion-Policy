"""
Test script for Force-Aware Diffusion Policy (FADP)
Tests the model with dummy inputs to verify implementation correctness.
"""

import sys
import os
import pathlib

# Add project root to path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

import torch
import numpy as np
from omegaconf import OmegaConf
from fadp.model.common.force_encoder import ForceEncoder
from fadp.policy.force_aware_diffusion_policy import ForceAwareDiffusionPolicy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model, model_name="Model"):
    """Print detailed model summary"""
    print(f"\n{'='*60}")
    print(f"{model_name} Structure")
    print(f"{'='*60}")
    print(model)
    
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"{model_name} Parameter Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Memory estimation (assuming float32)
    memory_mb = (total_params * 4) / (1024 ** 2)
    print(f"Estimated model size: {memory_mb:.2f} MB (float32)")
    print(f"{'='*60}\n")


def test_force_encoder():
    """Test ForceEncoder with dummy input"""
    print("\n" + "="*60)
    print("Testing ForceEncoder...")
    print("="*60)
    
    batch_size = 4
    n_obs_steps = 2
    force_dim = 6
    output_dim = 256
    hidden_dim = 512
    
    # Create encoder
    force_encoder = ForceEncoder(
        input_dim=force_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim
    )
    
    # Print model structure
    print_model_summary(force_encoder, "ForceEncoder")
    
    # Test single timestep
    print(f"\nTest 1: Single timestep force data")
    force_data = torch.randn(batch_size, force_dim)
    print(f"  Input shape: {force_data.shape}")
    
    output = force_encoder(force_data)
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {output_dim})")
    assert output.shape == (batch_size, output_dim), "Single timestep output shape mismatch!"
    print("  ✓ Passed!")
    
    # Test multiple timesteps
    print(f"\nTest 2: Multiple timestep force data")
    force_data_seq = torch.randn(batch_size, n_obs_steps, force_dim)
    print(f"  Input shape: {force_data_seq.shape}")
    
    output_seq = force_encoder(force_data_seq)
    print(f"  Output shape: {output_seq.shape}")
    print(f"  Expected: ({batch_size}, {n_obs_steps}, {output_dim})")
    assert output_seq.shape == (batch_size, n_obs_steps, output_dim), "Sequence output shape mismatch!"
    print("  ✓ Passed!")
    
    print(f"\nForceEncoder tests completed successfully!")
    return True


def test_policy_with_force():
    """Test ForceAwareDiffusionPolicy with force input"""
    print("\n" + "="*60)
    print("Testing ForceAwareDiffusionPolicy with Force Input...")
    print("="*60)
    
    # Configuration
    batch_size = 2
    n_obs_steps = 2
    n_action_steps = 8
    horizon = 16
    image_height = 96
    image_width = 96
    
    # Define shape_meta with force
    shape_meta = {
        'obs': {
            'camera_0': {
                'shape': [3, image_height, image_width],
                'type': 'rgb'
            },
            'force': {
                'shape': [6],
                'type': 'low_dim'
            }
        },
        'action': {
            'shape': [7]  # dx, dy, dz, drx, dry, drz, gripper
        }
    }
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Observation steps: {n_obs_steps}")
    print(f"  Action steps: {n_action_steps}")
    print(f"  Horizon: {horizon}")
    print(f"  Image size: {image_height}x{image_width}")
    print(f"  Action dim: 7 (dx, dy, dz, drx, dry, drz, gripper)")
    print(f"  Force dim: 6 (fx, fy, fz, mx, my, mz)")
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # Create policy
    print(f"\nCreating policy...")
    policy = ForceAwareDiffusionPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=8,
        obs_as_global_cond=True,
        crop_shape=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=True,
        eval_fixed_crop=False,
        force_encoder_hidden_dim=128
    )
    
    print(f"  ✓ Policy created successfully!")
    print(f"  Force encoder present: {policy.force_encoder is not None}")
    print(f"  RGB feature dim: {policy.obs_feature_dim}")
    print(f"  Force feature dim: {policy.force_feature_dim}")
    print(f"  Action dim: {policy.action_dim}")
    
    # Print detailed model structure
    print_model_summary(policy, "Complete FADP Policy")
    
    # Print component-wise parameter breakdown
    print(f"\n{'='*60}")
    print("Component-wise Parameter Breakdown")
    print(f"{'='*60}")
    
    components = [
        ("RGB Encoder (ResNet)", policy.obs_encoder),
        ("Force Encoder (MLP)", policy.force_encoder),
        ("Diffusion UNet", policy.model),
    ]
    
    for comp_name, comp in components:
        if comp is not None:
            total, trainable = count_parameters(comp)
            print(f"{comp_name:.<40} {total:>12,} ({total/1e6:>6.2f}M)")
    
    print(f"{'='*60}\n")
    
    # Create dummy observations
    print(f"\nCreating dummy observations...")
    obs_dict = {
        'camera_0': torch.rand(batch_size, n_obs_steps, 3, image_height, image_width),
        'force': torch.randn(batch_size, n_obs_steps, 6)  # Force/torque data
    }
    
    print(f"  Camera shape: {obs_dict['camera_0'].shape}")
    print(f"  Force shape: {obs_dict['force'].shape}")
    
    # Create dummy actions for training
    actions = torch.randn(batch_size, horizon, 7)
    print(f"  Actions shape: {actions.shape}")
    
    # Test normalization
    print(f"\nTesting normalizer...")
    from fadp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    from fadp.common.normalize_util import get_image_range_normalizer
    normalizer = LinearNormalizer()
    
    # Create dummy normalizers for images (use image range normalizer)
    normalizer['camera_0'] = get_image_range_normalizer()
    
    # Create normalizer for force data
    normalizer['force'] = SingleFieldLinearNormalizer.create_fit(
        torch.randn(100, 6)  # Dummy force data for normalization stats
    )
    
    # Create normalizer for actions
    normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
        torch.randn(100, 7)  # Dummy action data for normalization stats
    )
    
    policy.set_normalizer(normalizer)
    print(f"  ✓ Normalizer set successfully!")
    
    # Test forward pass (training)
    print(f"\nTest 1: Training loss computation...")
    batch = {
        'obs': obs_dict,
        'action': actions
    }
    
    try:
        loss = policy.compute_loss(batch)
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Loss shape: {loss.shape}")
        assert loss.shape == torch.Size([]), "Loss should be a scalar!"
        print(f"  ✓ Training loss computation passed!")
    except Exception as e:
        print(f"  ✗ Training loss computation failed: {e}")
        raise
    
    # Test inference
    print(f"\nTest 2: Inference (action prediction)...")
    policy.eval()
    
    with torch.no_grad():
        try:
            result = policy.predict_action(obs_dict)
            action_pred = result['action']
            
            print(f"  Predicted action shape: {action_pred.shape}")
            print(f"  Expected: ({batch_size}, {n_action_steps}, 7)")
            assert action_pred.shape == (batch_size, n_action_steps, 7), "Action prediction shape mismatch!"
            
            print(f"  Sample predicted actions (first timestep):")
            print(f"    dx: {action_pred[0, 0, 0].item():.4f}")
            print(f"    dy: {action_pred[0, 0, 1].item():.4f}")
            print(f"    dz: {action_pred[0, 0, 2].item():.4f}")
            print(f"    drx: {action_pred[0, 0, 3].item():.4f}")
            print(f"    dry: {action_pred[0, 0, 4].item():.4f}")
            print(f"    drz: {action_pred[0, 0, 5].item():.4f}")
            print(f"    gripper: {action_pred[0, 0, 6].item():.4f}")
            
            print(f"  ✓ Inference passed!")
        except Exception as e:
            print(f"  ✗ Inference failed: {e}")
            raise
    
    print(f"\nPolicy tests completed successfully!")
    return True


def test_policy_without_force():
    """Test backward compatibility: policy without force input"""
    print("\n" + "="*60)
    print("Testing Backward Compatibility (No Force Input)...")
    print("="*60)
    
    batch_size = 2
    n_obs_steps = 2
    n_action_steps = 8
    horizon = 16
    
    # Shape meta without force
    shape_meta = {
        'obs': {
            'camera_0': {
                'shape': [3, 96, 96],
                'type': 'rgb'
            }
        },
        'action': {
            'shape': [7]
        }
    }
    
    print(f"\nTesting policy WITHOUT force input...")
    print(f"  (Should still work for backward compatibility)")
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    policy = ForceAwareDiffusionPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=8,
        obs_as_global_cond=True,
        crop_shape=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    )
    
    print(f"  Force encoder present: {policy.force_encoder is not None}")
    print(f"  Force feature dim: {policy.force_feature_dim}")
    assert policy.force_encoder is None, "Force encoder should be None when no force in shape_meta"
    assert policy.force_feature_dim == 0, "Force feature dim should be 0 when no force"
    
    print(f"  ✓ Backward compatibility test passed!")
    return True


def test_multi_camera_with_force():
    """Test policy with multiple cameras and force"""
    print("\n" + "="*60)
    print("Testing Multi-Camera Setup with Force...")
    print("="*60)
    
    batch_size = 2
    n_obs_steps = 2
    n_action_steps = 8
    horizon = 16
    
    # Shape meta with multiple cameras
    shape_meta = {
        'obs': {
            'camera_0': {
                'shape': [3, 96, 96],
                'type': 'rgb'
            },
            'camera_1': {
                'shape': [3, 96, 96],
                'type': 'rgb'
            },
            'force': {
                'shape': [6],
                'type': 'low_dim'
            }
        },
        'action': {
            'shape': [7]
        }
    }
    
    print(f"\nTesting with 2 cameras + force sensor...")
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    policy = ForceAwareDiffusionPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=8,
        obs_as_global_cond=True,
        crop_shape=None,
    )
    
    # Create observations
    obs_dict = {
        'camera_0': torch.rand(batch_size, n_obs_steps, 3, 96, 96),
        'camera_1': torch.rand(batch_size, n_obs_steps, 3, 96, 96),
        'force': torch.randn(batch_size, n_obs_steps, 6)
    }
    
    print(f"  Camera 0 shape: {obs_dict['camera_0'].shape}")
    print(f"  Camera 1 shape: {obs_dict['camera_1'].shape}")
    print(f"  Force shape: {obs_dict['force'].shape}")
    
    # Print model summary for multi-camera setup
    print_model_summary(policy, "Multi-Camera FADP Policy")
    
    # Component breakdown
    print(f"\n{'='*60}")
    print("Component-wise Parameter Breakdown (Multi-Camera)")
    print(f"{'='*60}")
    
    components = [
        ("RGB Encoder (2x ResNet)", policy.obs_encoder),
        ("Force Encoder (MLP)", policy.force_encoder),
        ("Diffusion UNet", policy.model),
    ]
    
    for comp_name, comp in components:
        if comp is not None:
            total, trainable = count_parameters(comp)
            print(f"{comp_name:.<40} {total:>12,} ({total/1e6:>6.2f}M)")
    
    print(f"{'='*60}\n")
    
    # Setup normalizer
    from fadp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    from fadp.common.normalize_util import get_image_range_normalizer
    normalizer = LinearNormalizer()
    
    # Use image range normalizer for cameras
    for key in ['camera_0', 'camera_1']:
        normalizer[key] = get_image_range_normalizer()
    
    normalizer['force'] = SingleFieldLinearNormalizer.create_fit(torch.randn(100, 6))
    normalizer['action'] = SingleFieldLinearNormalizer.create_fit(torch.randn(100, 7))
    
    policy.set_normalizer(normalizer)
    
    # Test inference
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
        action_pred = result['action']
        
        print(f"  Predicted action shape: {action_pred.shape}")
        assert action_pred.shape == (batch_size, n_action_steps, 7)
        print(f"  ✓ Multi-camera test passed!")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("FORCE-AWARE DIFFUSION POLICY (FADP) - MODEL TESTS")
    print("="*60)
    
    tests = [
        ("ForceEncoder", test_force_encoder),
        ("Policy with Force", test_policy_with_force),
        ("Backward Compatibility", test_policy_without_force),
        ("Multi-Camera with Force", test_multi_camera_with_force),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, success, error in results:
        if success:
            print(f"✓ {test_name}: PASSED")
            passed += 1
        else:
            print(f"✗ {test_name}: FAILED")
            if error:
                print(f"  Error: {error}")
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed successfully!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

