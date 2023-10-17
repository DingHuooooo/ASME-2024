import os
import shutil
import torch.nn.functional as F
import numpy as np
import cv2
import torch
import random
import time
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
        input_img = (inputs[:3].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        target_mask = (targets[0].cpu().numpy() * 255).astype(np.uint8)
        overlaid = overlay_mask_on_image(input_img, target_mask)
        cv2.imwrite(os.path.join(path, f"{prefix}_overlay.png"), overlaid)

    for idx in range(2):
        # Randomly sample from the entire dataset
        random.seed(time.time())
        train_sample_idx = random.randint(0, len(train_dataloader.dataset) - 1)
        val_sample_idx = random.randint(0, len(val_dataloader.dataset) - 1)
        inputs, targets = train_dataloader.dataset[train_sample_idx]
        process_and_save(inputs, targets, f"train_{idx}")
        inputs, targets = val_dataloader.dataset[val_sample_idx]
        process_and_save(inputs, targets, f"val_{idx}")

    del inputs, targets, train_dataloader, val_dataloader, train_sample_idx, val_sample_idx
    torch.cuda.empty_cache()
    
    
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

def make_logger(path, local_rank=0):
    log_path = os.path.join(path, 'log.txt')

    class logger():
        def __init__(self, path):
            self.log_path = path

        def __call__(self, obj):
            print(obj)
            with open(self.log_path, 'a') as f:
                print(obj, file=f)
                
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(path, 'runs'))
    else:
        writer = None

    return logger(log_path), writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot