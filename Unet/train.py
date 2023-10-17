import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datasets.dataset import MyDataset
from models.unet import ResUnetPlusPlus, UNet
from utils import iou_loss, ensure_path, compute_num_params, save_random_image, make_logger, overlay_images, ce_loss

def check_model_parameters_across_processes(model, local_rank):
    params_list = [p.data.view(-1) for p in model.parameters()]
    params_tensor = torch.cat(params_list)

    world_size = dist.get_world_size()
    gathered_tensors = [torch.empty_like(params_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, params_tensor)

    if local_rank == 0:
        for i in range(1, world_size):
            if not torch.equal(gathered_tensors[0], gathered_tensors[i]):
                log(f"Process 0 and process {i} have different model parameters.")
                return False
        log("All processes have the same model parameters.")
        return True
    return None

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def create_dataloader(train_dir, gt_dir, need_transform, ratio=1.0):
    """Create dataloader with distributed support.""" 
    dataset = MyDataset(train_dir, gt_dir, need_transform=need_transform, ratio=ratio)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, sampler=sampler)
    del dataset, sampler
    return loader

def prepare_training():
    """Initialize model and optimizer for training."""
    # model = ResUnetPlusPlus(3)
    model = UNet(3, 1)
    model = model.cuda()
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=[local_rank],
        find_unused_parameters=False,
        broadcast_buffers=False
    )
    total_params = compute_num_params(model.module, text=True)
    if local_rank == 0:
        log(f'Total number of parameters in the model: {total_params}')
        if torch.cuda.device_count() > 1:
            log(f"Using {torch.cuda.device_count()} GPUs for training.")
    optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)
    return model, optimizer

def train(model, dataloader, optimizer, epoch):
    """Train model for one epoch."""
    model.train()
    train_loss = torch.tensor(0.0).to(device)
    train_iou = torch.tensor(0.0).to(device)
    train_iou_threshold = torch.tensor(0.0).to(device)

    # Progress bar only for the main process
    if local_rank == 0:
        train_progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Loss: N/A, IOU: N/A, IOU with Threshold: N/A], ", leave=False)
    else:
        train_progress_bar = dataloader

    # Training loop
    for batch in train_progress_bar:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        sigmoid = nn.Sigmoid()
        loss_bce = ce_loss(outputs, targets)
        loss_iou = iou_loss(sigmoid(outputs), targets)
        loss_iou_threshold = iou_loss((sigmoid(outputs) > 0.5).float(), targets)

        # Compute loss with optional L1/L2 regularization
        if need_L1L2:
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss_bce + loss_iou + l1_lambda * l1_norm
        else:
            loss = loss_bce + loss_iou

        loss.backward()
        optimizer.step()

        train_loss += loss
        train_iou += 1 - loss_iou
        train_iou_threshold += 1 - loss_iou_threshold

        # Update progress bar description
        if local_rank == 0:
            train_progress_bar.set_description(f"Epoch {epoch} [Loss: {loss.item():.4f} BCE Loss: {loss_bce.item():.4f} IOU: {1-loss_iou.item():.4f}, IOU with Threshold: {1-loss_iou_threshold.item():.4f}]")

    # Synchronize losses and metrics across processes
    dist.reduce(train_loss, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(train_iou, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(train_iou_threshold, dst=0, op=dist.ReduceOp.SUM)

    # Post-process for main process
    if local_rank == 0:
        train_loss /= dist.get_world_size() * len(dataloader)
        train_iou /= dist.get_world_size() * len(dataloader)
        train_iou_threshold /= dist.get_world_size() * len(dataloader)

    # Cleanup
    del inputs, targets, outputs, loss_bce, loss_iou, loss_iou_threshold, loss
    torch.cuda.empty_cache()

    return train_loss.item(), train_iou.item(), train_iou_threshold.item()

def validate(model, dataloader, epoch):
    """Validate model."""
    model.eval()
    val_loss = torch.tensor(0.0).to(device)
    val_iou = torch.tensor(0.0).to(device)
    val_iou_threshold = torch.tensor(0.0).to(device)

    # Progress bar only for the main process
    if local_rank == 0:
        val_progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Loss: N/A, IOU: N/A, IOU with Threshold: N/A], ", leave=False)
    else:
        val_progress_bar = dataloader

    # Validation loop
    with torch.no_grad():
        for val_batch in val_progress_bar:
            val_inputs, val_targets = val_batch
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            sigmoid = nn.Sigmoid()
            loss_bce = torch.tensor(0.0).to(device)
            loss_iou = torch.tensor(0.0).to(device)
            loss_iou_threshold = torch.tensor(0.0).to(device)
            for val_outoput, val_target in zip(val_outputs, val_targets):
                loss_bce += ce_loss(val_outoput, val_target)
                loss_iou += iou_loss(sigmoid(val_outoput), val_target)
                loss_iou_threshold += iou_loss((sigmoid(val_outoput) > 0.5).float(), val_target)

            loss_bce = loss_bce / len(val_outputs)
            loss_iou = loss_iou / len(val_outputs)
            loss_iou_threshold = loss_iou_threshold / len(val_outputs)
            

            # Compute loss with optional L1/L2 regularization
            if need_L1L2:
                l1_lambda = 1e-5
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss_val = loss_bce + loss_iou + l1_lambda * l1_norm
            else:
                loss_val = loss_bce + loss_iou

            val_loss += loss_val.item()
            val_iou += 1 - loss_iou.item()
            val_iou_threshold += 1 - loss_iou_threshold.item()

            # Update progress bar description
            if local_rank == 0:
                val_progress_bar.set_description(f"Epoch {epoch} [Loss: {loss_val.item():.4f}, BCE Loss: {loss_bce.item():.4f} IOU: {1-loss_iou.item():.4f}, IOU with Threshold: {1-loss_iou_threshold.item():.4f}]")

    # Synchronize losses and metrics across processes
    dist.reduce(val_loss, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(val_iou, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(val_iou_threshold, dst=0, op=dist.ReduceOp.SUM)

    # Post-process for main process
    if local_rank == 0:
        val_loss /= dist.get_world_size() * len(dataloader)
        val_iou /= dist.get_world_size() * len(dataloader)
        val_iou_threshold /= dist.get_world_size() * len(dataloader)

    # Cleanup
    del val_inputs, val_targets, val_outputs, loss_bce, loss_iou, loss_iou_threshold
    torch.cuda.empty_cache()

    return val_loss.item(), val_iou.item(), val_iou_threshold.item()

def main():
    """Main training loop."""
    model, optimizer = prepare_training()
    train_dataloader = create_dataloader(train_dir, train_gt_dir, need_transform=need_transform, ratio=1.0)
    val_dataloader = create_dataloader(val_train_dir, val_gt_dir, need_transform=False, ratio=1.0)
    if local_rank == 0:
        log(f"The size of the training is {train_dataloader.dataset[0][0].shape}, with number of images {len(train_dataloader.dataset)}")
        log(f"The size of the validation is {val_dataloader.dataset[0][0].shape}, with number of images {len(val_dataloader.dataset)}")
        save_random_image(train_dataloader, val_dataloader, save_path)

    best_val_loss = float('inf')

    # Epoch loop
    for epoch in range(1, epochs+1):
        train_dataloader.sampler.set_epoch(epoch)
        train_loss, train_iou, train_iou_threshold = train(model, train_dataloader, optimizer, epoch)

        torch.distributed.barrier()
        check_model_parameters_across_processes(model, local_rank)

        if local_rank == 0:
            writer.add_scalar('Loss/train_loss', train_loss, epoch)
            writer.add_scalar('IOU/train_iou', train_iou, epoch)
            writer.add_scalar('IOU/train_iou_threshold', train_iou_threshold, epoch)

        val_loss, val_iou, val_iou_threshold = validate(model, val_dataloader, epoch)

        if local_rank == 0:
            writer.add_scalar('Loss/val_loss', val_loss, epoch)
            writer.add_scalar('IOU/val_iou', val_iou, epoch)
            writer.add_scalar('IOU/val_iou_threshold', val_iou_threshold, epoch)

            log(f"Epoch {epoch}/{epochs}, Training Loss: {train_loss:.4f}, Training IOU: {train_iou:.4f}, Training IOU with Threshold: {train_iou_threshold:.4f} Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}, Validation IOU with Threshold: {val_iou_threshold:.4f}")

            # Save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), save_path + '/best_model.pth')
            if epoch in [5, 10, 15, 20, 40, 60, 80, 100]:
                torch.save(model.module.state_dict(), save_path + f'/model_{epoch}.pth')

if __name__ == '__main__':
    """Initialization and training process start."""
    # Distributed setup
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    os.environ["OMP_NUM_THREADS"] = "8"
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'

    set_seeds()

    # Load config
    root_url = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_url, 'configs/config.yaml')
    config = load_config(config_path)

    # Extract config parameters
    name = config['name']
    ratio = config['ratio']
    train_dir = config['train_dir']
    train_gt_dir = config['train_gt_dir']
    val_train_dir = config['val_train_dir']
    val_gt_dir = config['val_gt_dir']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    need_transform = config['need_transform']
    need_L1L2 = config['need_L1L2']

    # Modify model name based on configuration
    name = str(name) + '_L1L2' if need_L1L2 else str(name)
    name = str(name) + '_transform' if need_transform else str(name)
    name = str(name) + '_ratio_' + str(ratio)

    # Setup save path
    save_path = os.path.join(config['save_path'], name)

    if local_rank == 0:
        ensure_path(save_path)
        log, writer = make_logger(save_path)

    # Ensure all processes start the main training loop at the same time
    dist.barrier()

    main()

    # Clean up distributed process
    dist.destroy_process_group()
