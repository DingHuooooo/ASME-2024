import os
import json
import torch
import numpy as np
from PIL import Image
from utils import ensure_path, make_logger, iou_loss
from models.unet import UNet
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

def unet_predict(unet, image, gt):
    # predict with unet
    sigmoid = nn.Sigmoid()
    image_tensor = transforms.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0).cuda()
    outputs = unet(image_tensor)
    pre = (sigmoid(outputs)>0.5).float().squeeze().cpu().detach().numpy()
    iou = 1 - iou_loss(sigmoid(outputs), torch.from_numpy(gt/255).unsqueeze(0).unsqueeze(0).float().cuda()).item()
    iou_threshold = 1 - iou_loss((sigmoid(outputs) > 0.5).float(), torch.from_numpy(gt/255).unsqueeze(0).unsqueeze(0).float().cuda()).item()
    return pre, iou, iou_threshold

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用GPU 1
    save_path = "./save/UnetPredictions"
    ensure_path(save_path)
    logger, _ = make_logger(os.path.join(save_path))

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
        logger(f'Validation dataset has {len(images_list)} images of tool {tool}')

    # retrieve all models that are interested
    UNet_checkpoints = []
    model_names = [dir for dir in os.listdir('./save') if 'RGB' in dir]
    for model_name in model_names:
        for epochs in [5, 10, 15, 20, 40, 60, 80, 100]:    
            UNet_checkpoints.append(f"./save/{model_name}/model_{epochs}.pth")
            logger(f'{model_name} with {epochs} epochs will be tested.')

    # load model
    unet = UNet(3, 1)
    unet = unet.cuda()
    
    results = {} 
    for UNet_checkpoint in UNet_checkpoints:
        # load model checkpoint
        unet.load_state_dict(torch.load(UNet_checkpoint))
        unet.eval()

        model_name = os.path.basename(os.path.dirname(UNet_checkpoint))
        epoch = int(UNet_checkpoint.split("_")[-1].split(".")[0])

        if model_name not in results:
            results[model_name] = {}
        if epoch not in results[model_name]:
            results[model_name][epoch] = {}

        # load tools
        for tool in dataset.keys():
            logger(f"---{model_name}---Epoch{epoch}---{tool}---")
            iou_list = []
            ensure_path(os.path.join(save_path, model_name, str(epoch), tool))
            # load image lists and gt lists:
            for image_path, gt_path in zip(dataset[tool]['images'], dataset[tool]['gts']):
                image = np.array(Image.open(image_path).convert('RGB'))
                gt = np.array(Image.open(gt_path).convert('L'))

                # predict with unet
                unet_pre, iou, iou_threshold = unet_predict(unet, image, gt)

                # save the prediction
                Image.fromarray((unet_pre*255).astype(np.uint8)).save(os.path.join(save_path, model_name, str(epoch), tool, os.path.basename(image_path).replace("imaghe", "mask")))

                # Log the results
                logger(f"Model: {model_name}, Epoch: {epoch}, Tool: {tool}, Image: {os.path.basename(image_path)}, IOU_threshold: {iou_threshold}")
                iou_list.append(iou_threshold)

            # Compute average and standard deviation
            iou_avg = sum(iou_list) / len(iou_list)
            iou_std = np.std(iou_list)

            # Store average and standard deviation for the tool in the results
            results[model_name][epoch][tool] = {'avg': iou_avg, 'std': iou_std}

    # Print the final results with average and standard deviation
    logger('**************************')
    for model_name, epochs in results.items():
        for epoch, tools in epochs.items():
            total_iou = 0
            total_images = 0
            for tool, iou_data in tools.items():
                iou_avg = iou_data['avg']
                iou_std = iou_data['std']

                total_iou += iou_avg * len(dataset[tool]['images']) 
                total_images += len(dataset[tool]['images'])
                logger(f"Model: {model_name}, Epoch: {epoch}, Tool: {tool}, Avg IOU_threshold: {iou_avg}, Std IOU_threshold: {iou_std}")

            avg_iou_for_pair = total_iou / total_images
            logger(f"Model: {model_name}, Epoch: {epoch}, Overall Avg IOU_threshold: {avg_iou_for_pair}")
            logger(f'')

    # Save the results (with average and standard deviation) to a JSON file
    with open(os.path.join(save_path, "results.json"), "w") as outfile:
        json.dump(results, outfile)
