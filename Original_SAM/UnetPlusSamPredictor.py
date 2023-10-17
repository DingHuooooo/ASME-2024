import torch
import json
import numpy as np
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import pandas as pd
from models import sam_model_registry, SamPredictor
sys.path.append('../')
from Unet.models.unet import UNet
import matplotlib.pyplot as plt
import os
import gc
from PIL import Image
from torchvision import transforms
from utils import iou_loss, ensure_path, make_logger
from skimage.measure import find_contours
from points_finder import find_centroids_gravity_center_adjusted, find_centroids_shrink_mask, find_centroids_gravity_center_recursive


def convert_and_swap(input_data):
    if isinstance(input_data, tuple):
        # 如果输入是单个元组
        return [input_data[1], input_data[0]]
    elif isinstance(input_data, list):
        # 如果输入是元组的列表
        return [[item[1], item[0]] for item in input_data]
    elif isinstance(input_data, np.ndarray) and input_data.shape == (2,):
        # 如果输入是一个numpy数组形状为(2,)
        return [input_data[1], input_data[0]]
    else:
        raise TypeError(f"Expected input_data to be a tuple, list, or a numpy array of shape (2,), but got {type(input_data)}")

def extend_point(centroid, point, extension_length):
    # Compute the direction vector from the centroid to the point
    direction = point - centroid
    
    # Normalize the direction vector
    normalized_direction = direction / np.linalg.norm(direction)
    
    # Scale it by the extension length
    extended_vector = normalized_direction * extension_length
    
    # Compute the new extended point
    extended_point = point + extended_vector
    
    return extended_point


def unet_predict(unet, image, gt):
    # predict with unet
    sigmoid = nn.Sigmoid()
    image_tensor = transforms.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0).cuda()
    outputs = unet(image_tensor)
    outputs_lowres = nn.MaxPool2d(4)(outputs)
    pre = (sigmoid(outputs)>0.5).float().squeeze().cpu().detach().numpy()
    iou = 1 - iou_loss(sigmoid(outputs), torch.from_numpy(gt/255).unsqueeze(0).unsqueeze(0).float().cuda()).item()
    iou_threshold = 1 - iou_loss((sigmoid(outputs) > 0.5).float(), torch.from_numpy(gt/255).unsqueeze(0).unsqueeze(0).float().cuda()).item()
    del outputs, unet, image_tensor
    torch.cuda.empty_cache()
    return outputs_lowres, pre, iou, iou_threshold

def sam_predict(sam_predictor, image, gt, outputs_lowres, point_coords=None, point_labels=None):
    # predict with sam
    sam_predictor.set_image(image)
    masks, _, _ = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=outputs_lowres.squeeze(0).cpu().detach().numpy(),
        multimask_output=False,
    )
    plt.imshow(image)
    pre = masks
    iou_threshold = 1 - iou_loss((torch.from_numpy(masks).unsqueeze(0).float().cuda() > 0.5).float(), torch.from_numpy(gt/255).unsqueeze(0).unsqueeze(0).float().cuda()).item()
    sam_predictor.reset_image()
    del sam_predictor, image, gt, outputs_lowres
    torch.cuda.empty_cache()
    return pre, iou_threshold


# 只使用中心点
def find_points_1(pre_mask):

    # 获取这些连通区域的中心坐标
    _, _, centroids = find_centroids(pre_mask)

    centroids = convert_and_swap(centroids)

    input_point = np.array(centroids)
    input_label = np.array([1 for _ in range(len(input_point))])
    if len(input_point) > 0:
        return input_point, input_label
    else:
        return None, None

def find_points_2(unet_pre, extension_length=100):
    labeled_mask, large_region_labels, large_region_centroids = find_centroids(unet_pre)

    points_list = []
    label_list = []
    # find negative points
    padding = 5  # Specify the amount of padding

    for region_label, centroid in zip(large_region_labels, large_region_centroids):
        # Get the mask of the current region
        mask_for_region = labeled_mask == region_label
        
        # Pad the mask
        padded_mask = np.pad(mask_for_region, padding, mode='constant', constant_values=0)
        
        # Find contours on the padded mask
        contours = find_contours(padded_mask, 0.5)
        
        # Get the longest contour and adjust its coordinates due to padding
        longest_contour = sorted(contours, key=lambda x: len(x))[-1]
        longest_contour -= padding
        
        # Find points with the same x-coordinate as the centroid
        horizontal_points = [point for point in longest_contour if np.abs(point[1] - centroid[1]) < 1]
        
        # Find points with the same y-coordinate as the centroid
        vertical_points = [point for point in longest_contour if np.abs(point[0] - centroid[0]) < 2]
        
        # Get the top and bottom points by y-coordinate
        if horizontal_points:
            horizontal_points_sorted = sorted(horizontal_points, key=lambda x: x[0])
            bottom_point = horizontal_points_sorted[0]
            top_point = horizontal_points_sorted[-1]
            bottom_point = extend_point(centroid, horizontal_points_sorted[0], extension_length)
            top_point = extend_point(centroid, horizontal_points_sorted[-1], extension_length)
            bottom_point, top_point = convert_and_swap(bottom_point), convert_and_swap(top_point)
            points_list.extend([top_point, bottom_point])
            label_list.extend([0, 0])
        
        # Get the left and right points by x-coordinate
        if vertical_points:
            vertical_points_sorted = sorted(vertical_points, key=lambda x: x[1])
            left_point = vertical_points_sorted[0]
            right_point = vertical_points_sorted[-1]
            left_point = extend_point(centroid, vertical_points_sorted[0], extension_length)
            right_point = extend_point(centroid, vertical_points_sorted[-1], extension_length)
            left_point, right_point = convert_and_swap(left_point), convert_and_swap(right_point)
            points_list.extend([left_point, right_point])
            label_list.extend([0, 0])

        centroid = convert_and_swap(centroid)
        points_list.append(centroid)
        label_list.append(1)
    # Display the result
    input_point = np.array(points_list) if len(points_list) > 0 else None
    input_label = np.array(label_list) if len(label_list) > 0 else None
    return input_point, input_label

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU 0

    # find_centroids_methods = [find_centroids_gravity_center_recursive, find_centroids_shrink_mask, find_centroids_gravity_center_adjusted]
    find_centroids_methods = [find_centroids_shrink_mask]
    # Load model
    sam_checkpoint = "../src/sam_pretrained/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.cuda()
    predictor = SamPredictor(sam)
    # load model
    unet = UNet(3, 1)
    unet = unet.cuda()

    # retrieve all models that are interested
    UNet_checkpoints = []
    model_names = [dir for dir in os.listdir('../Unet/save') if 'RGB' in dir]
    for model_name in model_names:
        # for epochs in range(5, 105, 5):
        for epochs in [5, 10, 15, 20, 40, 60, 80, 100]:
            UNet_checkpoints.append(f"../Unet/save/{model_name}/model_{epochs}.pth")

    # retrieve val dataset
    dataset = {}
    val_image_path = "../src/dataset/Test/"
    val_gt_path = "../src/dataset/Test_gt/"
    for dir in os.listdir(val_image_path):
        tool = dir
        tool_image_path = os.path.join(val_image_path, tool)
        tool_gt_path = os.path.join(val_gt_path, tool)
        images_list = [os.path.join(tool_image_path, image_name) for image_name in os.listdir(tool_image_path)]
        gts_list = [os.path.join(tool_gt_path, image_name.replace("image", "label")) for image_name in os.listdir(tool_image_path)]
        dataset[tool] = {'images':images_list, 'gts':gts_list}
    

    for find_centroids in find_centroids_methods:
        results = {}  # This dictionary will store the results
        save_path = f"./save/UnetPlusSamPredictions/{find_centroids.__name__}/"
        ensure_path(save_path)
        logger = make_logger(os.path.join(save_path))
        # Ensure methods name is in results
        if find_centroids.__name__ not in results:
            results[find_centroids.__name__] = {}

        for UNet_checkpoint in UNet_checkpoints:
            # load model checkpoints
            unet.load_state_dict(torch.load(UNet_checkpoint))
            unet.eval()

            model_name = os.path.basename(os.path.dirname(UNet_checkpoint))
            epoch = int(UNet_checkpoint.split("_")[-1].split(".")[0])

            # Initialize model name if not already in results
            # for sam_input in ['Unet', 'SAM', 'SAMPlusCentroids', 'SAMPlusCentroidsPlusNegative']:
            for sam_input in ['Unet', 'SAMPlusCentroids', 'SAMPlusCentroidsPlusNegative']:
                if sam_input not in results[find_centroids.__name__]:
                    results[find_centroids.__name__][sam_input] = {}
                if model_name not in results[find_centroids.__name__][sam_input]:
                    results[find_centroids.__name__][sam_input][model_name] = {}
                if epoch not in results[find_centroids.__name__][sam_input][model_name]:
                    results[find_centroids.__name__][sam_input][model_name][epoch] = {}

            # load tools
            for tool in dataset.keys():
                logger(f"---{find_centroids.__name__}---{model_name}---Epoch{epoch}---{tool}---")
                iou_list_unet = []
                iou_list_sam = []
                iou_list_sam_centroids = []
                iou_list_sam_negative = []

                # load image lists and gt lists:
                for image_path, gt_path in zip(dataset[tool]['images'], dataset[tool]['gts']):
                    image = np.array(Image.open(image_path).convert('RGB'))
                    gt = np.array(Image.open(gt_path).convert('L'))

                    # predict with unet
                    outputs_lowres, unet_pre, _, iou_unet = unet_predict(unet, image, gt)
                    iou_list_unet.append(iou_unet)

                    # predict with sam
                    # _, iou_threshold_sam = sam_predict(predictor, image, gt, outputs_lowres)
                    #logger(f"Model: {model_name}, Epoch: {epoch}, Tool: {tool}, Image: {os.path.basename(image_path)}, IOU_threshold (SAM): {iou_threshold_sam}")
                    # iou_list_sam.append(iou_threshold_sam)

                    # predict with sam and centroids
                    input_point, input_label = find_points_1(unet_pre)
                    _, iou_threshold_sam_centroids = sam_predict(predictor, image, gt, outputs_lowres, input_point, input_label)
                    #logger(f"Model: {model_name}, Epoch: {epoch}, Tool: {tool}, Image: {os.path.basename(image_path)}, IOU_threshold (SAM + Centroids): {iou_threshold_sam_centroids}")
                    iou_list_sam_centroids.append(iou_threshold_sam_centroids)

                    # predict with sam and centroids and negative points
                    input_point, input_label = find_points_2(unet_pre)
                    _, iou_threshold_sam_negative = sam_predict(predictor, image, gt, outputs_lowres, input_point, input_label)
                    #logger(f"Model: {model_name}, Epoch: {epoch}, Tool: {tool}, Image: {os.path.basename(image_path)}, IOU_threshold (SAM + Centroids + Negative): {iou_threshold_sam_negative}")
                    iou_list_sam_negative.append(iou_threshold_sam_negative)

                # Store average IOU_threshold for the tool in the results
                # avg_iou_for_tool_sam = sum(iou_list_sam) / len(iou_list_sam)
                avg_iou_for_tool_sam_centroids = sum(iou_list_sam_centroids) / len(iou_list_sam_centroids)
                avg_iou_for_tool_unet = sum(iou_list_unet)/len(iou_list_unet)
                avg_iou_for_tool_sam_negative = sum(iou_list_sam_negative) / len(iou_list_sam_negative)

                std_iou_for_tool_unet = np.std(iou_list_unet)
                std_iou_for_tool_sam_negative = np.std(iou_list_sam_negative)
                std_iou_for_tool_sam_centroids = np.std(iou_list_sam_centroids)
                results[find_centroids.__name__]['Unet'][model_name][epoch][tool] = {'avg':avg_iou_for_tool_unet, 'std':std_iou_for_tool_unet}
                # results[find_centroids.__name__]['SAM'][model_name][epoch][tool] = {'avg':avg_iou_for_tool_sam, 'std':avg_iou_for_tool_sam}
                results[find_centroids.__name__]['SAMPlusCentroids'][model_name][epoch][tool]= {'avg':avg_iou_for_tool_sam_centroids, 'std':std_iou_for_tool_sam_centroids}
                results[find_centroids.__name__]['SAMPlusCentroidsPlusNegative'][model_name][epoch][tool] = {'avg':avg_iou_for_tool_sam_negative, 'std':std_iou_for_tool_sam_negative}
                
                # logger(f"Methods: {find_centroids.__name__}, Model: {model_name}, Epoch: {epoch}, Tool: {tool}, Avg IOU_threshold (Unet): {avg_iou_for_tool_unet}")
                # logger(f"Methods: {find_centroids.__name__}, Model: {model_name}, Epoch: {epoch}, Tool: {tool}, Avg IOU_threshold (SAM): {avg_iou_for_tool_sam}")
                # logger(f"Methods: {find_centroids.__name__}, Model: {model_name}, Epoch: {epoch}, Tool: {tool}, Avg IOU_threshold (SAM + Centroids): {avg_iou_for_tool_sam_centroids}")
                # logger(f"Methods: {find_centroids.__name__}, Model: {model_name}, Epoch: {epoch}, Tool: {tool}, Avg IOU_threshold (SAM + Centroids + Negative): {avg_iou_for_tool_sam_negative}")

                del outputs_lowres, unet_pre
                torch.cuda.empty_cache()
                gc.collect()
        # Save the results to a JSON file
        with open(f"./save/UnetPlusSamPredictions/{find_centroids.__name__}/results.json", "w") as outfile:
            json.dump(results, outfile)

