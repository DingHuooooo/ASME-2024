import argparse
import os
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

import copy
import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist

# Get the local rank from the environment variable
local_rank = int(os.getenv('LOCAL_RANK', '0'))

# Then use local_rank as before
torch.distributed.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
os.environ["OMP_NUM_THREADS"] = "4"

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    # instantiate the dataset of paired-image-folders of train and val
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    # instantiate the dataset of wrapper of train and val
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            # k,v = 'inp', self.img_transform(img); 'gt', self.mask_transform(mask)
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=16, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    pre_loader = make_data_loader(config.get('pre_dataset'), tag='pre')
    return train_loader, val_loader, pre_loader


def prepare_training():
    if config.get('PYTORCH_CUDA_ALLOC_CONF') is not None:
        max_split_size_mb = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        if local_rank == 0:
            log(f"PYTORCH_CUDA_ALLOC_CONF:{max_split_size_mb}")
            config['PYTORCH_CUDA_ALLOC_CONF'] = max_split_size_mb
    else:
        pass
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        # The calc_f1 function calculates the F1-score and Area Under the ROC Curve (AUC) 
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        # This function calc_fmeasure calculates the F-measure and the Mean Absolute Error (MAE)
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        # The function calc_ber calculates the Balanced Error Rate (BER) for binary classification tasks
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        # The calc_cod function is used to compute several evaluation metrics
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'iou':
        # The calc_cod function is used to compute calc_iou_mae_f1
        metric_fn = utils.calc_iou
        metric1, metric2, metric3, metric4 = 'iou', 'iou_thresholded', 'none', 'none'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()
        
        inp = batch['inp'] # inp = torch.Size([batch_size, 3, 1024, 1024])
        with torch.no_grad():
            pred = model.infer(inp)
        # print(pred.max(), pred.min())
        # tensor(-0.6077, device='cuda:1', grad_fn=<MaxBackward1>) tensor(-1.1844, device='cuda:1', grad_fn=<MinBackward1>)
        pred = torch.sigmoid(pred) # pred = torch.Size([batch_size, 1, 1024, 1024])
        
        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]
    
        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)
        
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)
    
    # Delete variables and free up memory before the prediction
    del batch, pred_list, gt_list, batch_pred, batch_gt
    torch.cuda.empty_cache()
    
    return result1, result2, result3, result4, metric1, metric2, metric3, metric4

def predict(loader, model, pre_ratio=1):
    model.eval()

    pred_list = []
    gt_list = []
    inp_list = []
    pre_loader = []
    number_of_samples = int(-(-len(loader) * pre_ratio // 1))

    for i, batch in enumerate(loader):
        if i == number_of_samples:
            break
        pre_loader.append(batch)
    
    if local_rank == 0:
        pbar = tqdm(total=len(pre_loader), leave=False, desc='pre')
    else:
        pbar = None

    for batch in pre_loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp'] # inp = torch.Size([batch_size, 3, 1024, 1024])
        with torch.no_grad():
            pred = model.infer(inp)
        pred = torch.sigmoid(pred) # pred = torch.Size([batch_size, 1,1024, 1024])
        pred = (pred >= 0.5).to(torch.int)

        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]
        batch_inp = [torch.zeros_like(batch['inp']) for _ in range(dist.get_world_size())]

        # print(pred.shape, batch['gt'].shape, batch['inp'].shape)
        # torch.Size([1, 1, 1024, 1024]) torch.Size([1, 1, 1024, 1024]) torch.Size([1, 3, 1024, 1024])
        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)
        dist.all_gather(batch_inp, batch['inp'])
        inp_list.extend(batch_inp)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 0) 
    gt_list = torch.cat(gt_list, 0) 
    inp_list = torch.cat(inp_list, 0) 

    # print(pred_list.shape, gt_list.shape, inp_list.shape)
    # torch.Size([len(pre_loader), 1, 1024, 1024]) torch.Size([len(pre_loader), 1, 1024, 1024]) torch.Size([len(pre_loader), 3, 1024, 1024])
    
    # Inverse input images normalization
    if loader.dataset.inverse_transform is not None:
        inverse_transform = inverse_transform = loader.dataset.inverse_transform
        for i in range(len(inp_list)):
            inp_list[i] = inverse_transform(inp_list[i])
            
    # Binarization
    for i in range(len(pred_list)):
        pred_list[i] = (pred_list[i] >= 0.5).to(torch.int)

    # Convert tensors to numpy arrays
    pred_list_np = pred_list.cpu().numpy()
    gt_list_np = gt_list.cpu().numpy()
    inp_list_np = inp_list.cpu().numpy()

    # Convert to RGB format
    inp_list_np = np.transpose(inp_list_np, (0, 2, 3, 1)) # if your images are in the format BxCxHxW
    inp_list_np = (inp_list_np * 255).astype(np.uint8)
    # Convert to Greyscale format
    gt_list_np = np.squeeze(gt_list_np, axis=1)
    pred_list_np = np.squeeze(pred_list_np, axis=1)
    # print(pred_list_np.shape, gt_list_np.shape, inp_list_np.shape)
    #(len(pre_loader), 1024, 1024) (len(pre_loader), 1024, 1024) (len(pre_loader), 1024, 1024, 3)
    predictions = []
    for i in range(len(dist.get_world_size()*pre_loader)):
        predictions.append({"inp": inp_list_np[i], "gt": gt_list_np[i], "pred": pred_list_np[i]})
        
    # Delete variables and free up memory after the prediction
    del batch, pred_list, gt_list, inp_list, batch_pred, batch_gt, batch_inp, pre_loader
    torch.cuda.empty_cache()

    return predictions


def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_all_list = []
    loss_IoU_list = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss_all = [torch.zeros_like(model.loss_all) for _ in range(dist.get_world_size())]
        batch_loss_IoU = [torch.zeros_like(model.loss_IoU) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss_all, model.loss_all)
        dist.all_gather(batch_loss_IoU, model.loss_IoU)
        loss_all_list.extend(batch_loss_all)
        loss_IoU_list.extend(batch_loss_IoU)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss_all = [i.item() for i in loss_all_list]
    loss_IoU = [i.item() for i in loss_IoU_list]
    
    # Delete variables and free up memory before the evaluation
    del batch, loss_all_list, loss_IoU_list, batch_loss_all, batch_loss_IoU
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary())
    return mean(loss_all), mean(loss_IoU)


def main(config_, save_path, args):
    
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    
    if config.get('data_norm') is None:
        config['data_norm'] = {'inp': {'sub': [0, 0, 0], 'div': [1, 1, 1]}}
    else:
        data_norm = config.get('data_norm')
        config['train_dataset']['wrapper']['args']['data_norm'] = copy.deepcopy(data_norm)
        config['val_dataset']['wrapper']['args']['data_norm'] = copy.deepcopy(data_norm)
        config['pre_dataset']['wrapper']['args']['data_norm'] = copy.deepcopy(data_norm)
      
    train_loader, val_loader, pre_loader = make_data_loaders()
    # construct model, optimizer, epoch_start, lr_scheduler
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    # why schedule lr again?
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        # set to False because all the parameters are used?
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    # Load checkpoint 
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    # calculate the number of parameters
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_all, train_loss_IoU = train(train_loader, model)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train: loss_all={:.4f} loss_IoU={:.4f}'.format(train_loss_all, train_loss_IoU))
            writer.add_scalars('loss', {'loss_all': train_loss_all, 'loss_IoU': train_loss_IoU}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model, eval_type=config.get('eval_type'))
            # Predictions and saving the images
            predictions = predict(pre_loader, model, pre_ratio=1)
            save_images(predictions, save_path, epoch)
            # Delete variables and free up memory after the prediction
            del predictions
            torch.cuda.empty_cache()

            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save(config, model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()

def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))

def save_images(predictions, save_path, epoch):
    base_path = os.path.join(save_path, f"predictions/model_epoch_{epoch}")
    os.makedirs(base_path, exist_ok=True)
    
    for i, prediction in enumerate(predictions):
        # Get numpy arrays
        inp_img = prediction["inp"]
        pred_img = prediction["pred"]
        gt_img = prediction["gt"]

        # Convert numpy array to PIL Image
        inp_img = Image.fromarray(inp_img)
        pred_img = Image.fromarray((pred_img * 255).astype(np.uint8), 'L')
        gt_img = Image.fromarray((gt_img * 255).astype(np.uint8), 'L')
        
        # Save images
        inp_img.save(os.path.join(base_path, f"inp_{i}.png"))
        pred_img.save(os.path.join(base_path, f"pred_{i}.png"))
        gt_img.save(os.path.join(base_path, f"gt_{i}.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/iou-sam-vit-l-AVM2022.yaml")
    parser.add_argument('--name', default="iou-sam-vit-l-AVM2022")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')
            
    save_tag = str(config.get('tag'))
    save_name = args.name
    
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    
    save_name += '_' + save_tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 train.py