
import h5py
import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F

def test_crop():
    # Path to a sample file
    data_path = 'data/basin_noup/episode0.hdf5'
    
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    try:
        with h5py.File(data_path, 'r') as f:
            print("Keys in HDF5:", list(f.keys()))
            
            # Assuming standard structure, try to find image data
            # Adjust keys based on actual structure. Often under 'observations/images/...'
            if 'observations' in f:
                obs = f['observations']
                if 'images' in obs:
                    images = obs['images']
                    print("Image keys:", list(images.keys()))
                    # Pick the first camera
                    cam_key = list(images.keys())[0]
                    img_data = images[cam_key][0] # Get first frame
                else:
                    # Maybe flat structure?
                    print("Obs keys:", list(obs.keys()))
                    # Try to find something that looks like an image
                    for k in obs.keys():
                        if 'rgb' in k or 'image' in k:
                            img_data = obs[k][0]
                            cam_key = k
                            break
            else:
                 # Check root keys for images
                 for k in f.keys():
                        if 'rgb' in k or 'image' in k:
                            img_data = f[k][0]
                            cam_key = k
                            break
                            
        print(f"Loaded image from {cam_key}, shape: {img_data.shape}")
        
        # Convert to torch tensor (C, H, W) float [0,1]
        # Assuming HWC uint8 input
        img_tensor = torch.from_numpy(img_data).permute(2, 0, 1).float() / 255.0
        
        # Simulate the logic in FADPEncoder
        # Assume model expects 224x224 input
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        H_in, W_in = img_tensor.shape[-2:]
        print(f"Input size (resized to model input): {H_in}x{W_in}")
        
        # Save original (resized)
        cv2.imwrite('test_crop_original.jpg', cv2.cvtColor((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        if H_in == W_in:
            # Use a more aggressive crop ratio to remove black borders
            crop_ratio = 0.5  # More aggressive than 0.707
            crop_size = int(H_in * crop_ratio)
            start = (H_in - crop_size) // 2
            print(f"Crop ratio: {crop_ratio}, Crop size: {crop_size}, Start index: {start}")
            
            img_cropped = img_tensor[..., start:start+crop_size, start:start+crop_size]
            
            # Save cropped
            cv2.imwrite('test_crop_cropped.jpg', cv2.cvtColor((img_cropped.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            # Resize to 224x224
            img_resized = F.interpolate(img_cropped.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
             # Save resized
            cv2.imwrite('test_crop_resized.jpg', cv2.cvtColor((img_resized.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            print("Saved: test_crop_original.jpg, test_crop_cropped.jpg, test_crop_resized.jpg")
        else:
            print("Image is not square, skipping crop test.")

    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crop()

