import torch
import numpy as np
import cv2
import os
import shutil

def overlay_images(inputs, outputs):
    """Overlay masks on images using OpenCV."""
    overlaid_images = []
    
    for image, mask in zip(inputs, outputs):
        # Convert tensors to numpy arrays and then scale to range [0, 255]
        image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask = (mask.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        mask_rgb = np.zeros_like(image)
        mask_rgb[:,:,0] = mask  # Set only the red channel
        
        # Overlay mask on image
        overlaid_image = cv2.addWeighted(image, 0.8, mask_rgb, 0.2, 0)
        
        # Convert back to tensor and append to the list
        overlaid_images.append(torch.from_numpy(overlaid_image).permute(2, 0, 1))
    
    # Stack the list of tensors to a single tensor
    return torch.stack(overlaid_images)

def iou_loss(pred, target):
    # Check the channel dimension
    if pred.dim() == 4:
        # Calculate IOU for each sample in the batch and then average
        intersection = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
        iou = (intersection + 1e-8) / (union + 1e-8)
        return 1 - iou.mean()
    elif pred.dim() == 3:
        # Original IOU calculation for the entire batch
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + 1e-8) / (union + 1e-8)
        return 1 - iou
    else:
        raise ValueError("Channel dimension must be 3 or 4.")
    

def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove:
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError as e:
                print(f"Error: {path} : {e}")
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

def make_logger(path):
    log_path = os.path.join(path, 'log.txt')

    class logger():
        def __init__(self, path):
            self.log_path = path

        def __call__(self, obj):
            print(obj)
            with open(self.log_path, 'a') as f:
                print(obj, file=f)

    return logger(log_path)