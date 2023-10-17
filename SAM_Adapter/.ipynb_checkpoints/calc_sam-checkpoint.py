import yaml
import torch
from tqdm import tqdm
from models import models


def batch_pred(loader, model):
    model.eval()
    preds = []
    pbar = tqdm(loader, leave=False, desc='val')

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        with torch.no_grad():
            pred = torch.sigmoid(model.infer(inp))
        preds.append(pred.cpu())
    return preds

def config_modify(config):
    import copy
    if config.get('data_norm') == 'None':
        config['data_norm'] = {'inp': {'sub': [0, 0, 0], 'div': [1, 1, 1]}}
    else:
        pass
    
    data_norm = config.get('data_norm')
    config['test_dataset']['wrapper']['args']['data_norm'] = copy.deepcopy(data_norm)
    
    return config

from PIL import Image
import torch
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
import cv2


def process_image(image, filename):
    # Crop the bottom part to make the height 1024
    if image.height > 1024:
        image = image.crop((0, image.height - 1024, image.width, image.height))

    # Padding width to 1024 * 5
    if image.width < 1024 * 5:
        padded_image = Image.new("RGB", (1024 * 5, 1024), color="black")
        padded_image.paste(image, (0, 0))
        image = padded_image

    #filename = os.path.join(output_directory, "patch_" + str(filename) + ".png") # Add the .png extension
    #image_array = np.array(image)
    #cv2.imwrite(filename, image_array)

    return image



# Extract patches
def extract_patches(image, patch_size=1024):
    patches = []
    for i in range(0, image.height, patch_size):
        for j in range(0, image.width, patch_size):
            patch = image.crop((j, i, j + patch_size, i + patch_size))
            patches.append(patch)
    return patches


# Predict patches
def predict_patches(patches, model, device, filename):
    masks = []
    i=0
    for patch in patches:
        patch_tensor = transforms.ToTensor()(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            mask = model.infer(patch_tensor)
            mask = torch.sigmoid(mask)
            #mask = (mask >= 0.1).float()
            mask_ = mask.cpu().squeeze().numpy().astype(np.uint8)
            #i+=1
            #filename = os.path.join(output_directory, "patch_" + str(filename) + str(i) + ".png") # Add the .png extension
            #cv2.imwrite(filename, mask_)
        masks.append(mask)
    return masks


def reassemble_masks(masks, original_image_size, device):
    reassembled_mask = torch.zeros((1, 1024, 1024 * 5), device=device)
    idx = 0
    for j in range(0, reassembled_mask.size(2), 1024):
        reassembled_mask[0, :, j:j+1024] = masks[idx][0].clone().detach().to(device)
        idx += 1

    # Padding height on the top
    padded_height = torch.zeros((1, original_image_size[1] - 1024, 1024 * 5), device=device)
    padded_reassembled_mask = torch.cat([padded_height, reassembled_mask], dim=1)

    # Cropping width to original size
    full_mask = padded_reassembled_mask[:, :, :original_image_size[0]]

    return full_mask


import numpy as np

def overlay_mask(image, full_mask, overlay_color=(255, 255, 0)):
    mask_np = (full_mask*255).cpu().numpy().squeeze().astype(np.uint8)

    image_np = np.array(image)  # Convert the PIL Image to a NumPy array
    image_with_mask = image_np.copy()

    for c in range(3):  # 3 channels for RGB
        image_with_mask[:, :, c] = image_with_mask[:, :, c] * (1 - mask_np / 255.0) + overlay_color[c] * (mask_np / 255.0)

    image_with_mask = Image.fromarray(image_with_mask.astype(np.uint8))  # Convert back to PIL Image

    return image_with_mask


def find_max_rotated_rect(full_mask: torch.Tensor, y_threshold: float, area_threshold: float = 1000) -> list:
    full_mask_np = (full_mask*255).cpu().numpy().squeeze().astype(np.uint8)
    contours, _ = cv2.findContours(full_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes_and_contours = []
    height_threshold = full_mask_np.shape[0] - y_threshold

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Calculate the area of the bounding box
        area = cv2.contourArea(box)

        # Check if the lowest y-coordinate of the box is not below the threshold and area is greater than the area threshold
        if min(pt[1] for pt in box) >= height_threshold and area > area_threshold:
            boxes_and_contours.append((box, contour))

    return boxes_and_contours


def calculate_values_for_box(box, full_mask):
    # Calculate the y-difference for the box
    y_diff = np.max([pt[1] for pt in box]) - np.min([pt[1] for pt in box])

    # Convert box points to integer
    box = np.int0(box)

    # Find the bounding rectangle of the box
    x, y, w, h = cv2.boundingRect(box)

    # Crop the full_mask using the bounding rectangle
    cropped_full_mask = full_mask[y:y+h, x:x+w].squeeze().cpu().numpy()

    # Create a mask for the box
    box_mask = np.zeros_like(full_mask.squeeze().cpu().numpy())
    cv2.drawContours(box_mask, [box], -1, 1, thickness=cv2.FILLED)

    # Find the overlapped area by multiplying the cropped mask with the box mask
    overlapped_area = cropped_full_mask * box_mask[y:y+h, x:x+w]

    # Count the number of pixels that have a value of 1 in the overlapped area
    mask_pixel_count = np.sum(overlapped_area == 1)

    return y_diff, mask_pixel_count, overlapped_area




if __name__ == '__main__':
    import os
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    os.environ["OMP_NUM_THREADS"] = "8"

    config_path = "/home/de532237/cancer/SAM/SAM_Adapter/configs/iou-sam-vit-b-test.yaml"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = config_modify(config)

    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(config['sam_checkpoint'], map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    model.eval()
    # Parameters
    image_directory = "/home/de532237/cancer/SAM/test"
    output_directory = "/home/de532237/cancer/SAM/test/predictions/"

    # Process images
    for filename in tqdm(os.listdir(image_directory), desc="Processing Images"):
        if filename.endswith(".jpg"):  # Adjust for other image formats if needed
            image_path = os.path.join(image_directory, filename)
            image = Image.open(image_path)
            original_image_size = image.size

            # Pre-processing
            processed_image = process_image(image, filename)

            # Extract patches
            patches = extract_patches(processed_image)

            # Predict patches
            masks = predict_patches(patches, model, device, filename)

            # Reassemble full-size mask
            full_mask = reassemble_masks(masks, original_image_size, device)

            # Overlay full_mask onto image
            image_with_mask_overlayed = overlay_mask(image, full_mask)
            image_with_mask_overlayed_np = np.array(image_with_mask_overlayed) # Convert to NumPy array
            cv2.imwrite(os.path.join(output_directory, "maskoverlayed_" + filename), image_with_mask_overlayed_np)


            # Find the maximum rotated rectangle
            y_threshold = 256  # Example threshold value
            area_threshold = 1000
            boxes_and_contours = find_max_rotated_rect(full_mask, y_threshold, area_threshold)

            # Create a copy of the original image to draw on
            image_with_mask = np.copy(image_with_mask_overlayed_np)
            
            # Convert the PIL image to a NumPy array
            image_np = np.array(image)

            # Loop through the boxes and contours
            for i, (box, contour) in enumerate(boxes_and_contours):
                # Draw contours on the original image
                image_np = cv2.drawContours(image_np, [box], 0, (0, 255, 0), 3)
                image_np = cv2.drawContours(image_np, [contour], 0, (255, 0, 0), 2)

                # Calculate the y-difference and mask pixel count for the current box
                y_diff, mask_pixel_count, overlapped_area = calculate_values_for_box(box, full_mask)

                # Define the text position based on the box position
                text_position = (int(box[:, 0].min() + 10), int(box[:, 1].min() - 10))

                # Write the values on the image, including the box index
                text = f"Box {i}: Height Diff: {y_diff:.2f}, Pixels: {mask_pixel_count}"
                cv2.putText(image_np, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Save the image
            cv2.imwrite(os.path.join(output_directory, "info_" + filename), image_np)


    
    
    
    