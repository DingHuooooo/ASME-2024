import os
import shutil
import torch.nn.functional as F
import numpy as np
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

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
    

def save_random_image(train_dataloader, val_dataloader, path):
    def overlay_mask_on_image(image, mask, alpha=0.2):
        """Overlay mask on image using cv2"""
        mask_rgb = np.zeros_like(image)
        mask_rgb[:,:,0] = mask  # Set only red channel
        return cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    
    # Function to process and save the overlaid image
    def process_and_save(inputs, targets, prefix):
        idx = np.random.randint(inputs.size(0))
        input_img = (inputs[idx][:3].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        target_mask = (targets[idx][0].cpu().numpy() * 255).astype(np.uint8)
        overlaid = overlay_mask_on_image(input_img, target_mask)
        cv2.imwrite(os.path.join(path, f"{prefix}_overlay.png"), overlaid)

    # Process for train
    for idx in range(2):
        inputs, targets = next(iter(train_dataloader))
        process_and_save(inputs, targets, f"train_{idx}")

        # Process for validation
        inputs, targets = next(iter(val_dataloader))
        process_and_save(inputs, targets, f"val_{idx}")
    
    
def iou_loss(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    return 1 - iou

def ce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)
    
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
    log = logger(log_path)

    writer = SummaryWriter(os.path.join(path, 'runs'))
    return log, writer

def compute_num_params(params_count, text=False):
    tot = int(params_count)
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot
