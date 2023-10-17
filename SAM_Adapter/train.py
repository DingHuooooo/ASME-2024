import argparse
import torch
import torch.distributed as dist
import os
import shutil
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from statistics import mean
import copy

import datasets
import models
import utils

DEBUG = False

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    # instantiate the dataset of paired-image-folders of train and val
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    # instantiate the dataset of wrapper of train and val
    if local_rank == 0:
        log('\n{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            if k != 'filename':
                log('  {}: shape={}'.format(k, tuple(v.shape)))

    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=16, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    model = models.make(config['model']).cuda()
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=[local_rank],
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)

    frozen_parts = set()
    unfrozen_parts = set()

    for name, para in model.named_parameters():
        main_parts = ".".join(name.split(".")[:2])

        if ("image_encoder" in name and "adaptor" not in name):
            para.requires_grad_(False)
            frozen_parts.add(main_parts)
        else:
            unfrozen_parts.add(main_parts)

    if local_rank == 0:
        log("\nFrozen parts:")
        for part in frozen_parts:
            log(part)

        log("\nUnfrozen parts:")
        for part in unfrozen_parts:
            log(part)


    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    model.optimizer = optimizer
    epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))

    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_total_params = utils.compute_num_params(model_total_params, text=True)
        model_grad_params = utils.compute_num_params(model_grad_params, text=True)

        log('\nmodel_grad_params:' + str(model_grad_params) + '\nmodel_total_params:' + str(model_total_params))
    
    return model, optimizer, epoch_start, lr_scheduler

def config_modify(config):
    if config.get('PYTORCH_CUDA_ALLOC_CONF') is not None:
        max_split_size_mb = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        if local_rank == 0:
            log(f"PYTORCH_CUDA_ALLOC_CONF:{max_split_size_mb}")
            config['PYTORCH_CUDA_ALLOC_CONF'] = max_split_size_mb
    else:
        pass
    
    for dataset in ['train_dataset', 'val_dataset', 'test_dataset']:
        config[dataset]['dataset']['args']['root_path_1'] = config[dataset]['dataset']['args']['root_path_1'].replace('datasetname', config['dataset_name'])
        config[dataset]['dataset']['args']['root_path_2'] = config[dataset]['dataset']['args']['root_path_2'].replace('datasetname', config['dataset_name'])
    
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config

def evaluate(loader, model, eval_type=None):
    model.eval()
    
    if eval_type == 'iou':
        metric_fn = utils.calc_iou
        metric1, metric2, metric3, metric4 = 'iou', 'iou_thresholded', 'mae', 'none'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    mask_list = []
    for batch in loader:
        for k, v in batch.items():
            if k!='filename':
                batch[k] = v.cuda()
        with torch.no_grad():
            pred = model.infer(batch['image'])
            pred = torch.sigmoid(pred) # pred = torch.Size([batch_size, 1, 1024, 1024])
        
        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_mask = [torch.zeros_like(batch['mask']) for _ in range(dist.get_world_size())]
    
        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_mask, batch['mask'])
        mask_list.extend(batch_mask)
        
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 1)
    mask_list = torch.cat(mask_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, mask_list)
    
    del batch, pred_list, mask_list, batch_pred, batch_mask
    torch.cuda.empty_cache()
    
    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


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
            if k!='filename':
                batch[k] = v.to(device)
        model.set_input(batch['image'], batch['mask'])
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
    
    del batch, loss_all_list, loss_IoU_list, batch_loss_all, batch_loss_IoU
    torch.cuda.empty_cache()
    return mean(loss_all), mean(loss_IoU)


def main(config_, save_path):
    global config, log, writer, log_info
    config = config_modify(config_)
    log, writer = utils.set_save_path(save_path, remove=False)
    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = 1e8
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_all, train_loss_IoU = train(train_loader, model)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            log_info.append('train: loss_all={:.4f} loss_IoU={:.4f}'.format(train_loss_all, train_loss_IoU))
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalars('loss', {'loss_all': train_loss_all, 'loss_IoU': train_loss_IoU}, epoch)
            save(model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = evaluate(val_loader, model, eval_type=config.get('eval_type'))
            if local_rank == 0:
                results = [result1, result2, result3, result4]
                metrics = [metric1, metric2, metric3, metric4]
                for metric, result in zip(metrics, results):
                    log_info.append('val: {}={:.4f}'.format(metric, result))
                    writer.add_scalars(metric, {'val': result}, epoch)

                if result3 < max_val_v:
                    max_val_v = result3
                    save(model, save_path, 'best')
                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
                log(', '.join(log_info))
                writer.flush()

def save(model, save_path, name):
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
                
if __name__ == '__main__':

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    os.environ["OMP_NUM_THREADS"] = "8"

    config_path = "configs/iou-sam-vit-b.yaml"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')
            
    save_prefix = str(config.get('sam_prefix'))
    save_postfix = str(config.get('modeltype'))
    save_name = save_prefix + '-' + config.get('dataset_name') + '_' + save_postfix
    if DEBUG:
        save_name = '_debug_' + save_name
    save_path = os.path.join('./save', save_name)
    if local_rank == 0:
        utils.ensure_path(save_path)
        print('save path : {} is cleared.'.format(save_path))
    torch.distributed.barrier()

    main(config, save_path)


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 train.py

