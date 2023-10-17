import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import models   # Assuming YourModel is your model class
import datasets  # Assuming datasets is your data processing module
import utils
import numpy as np
from typing import Optional, Tuple
from PIL import Image

class Predictor:
    def __init__(self, model):
        self.model = model
        self.input_image_torch = None
        self.features = None
        self.mask_threshold = 0.0

    def set_image(self, input_image: np.ndarray) -> None:
        if self.input_image_torch is not None:
            del self.input_image_torch, self.features
            torch.cuda.empty_cache()
            
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        # Check if the input image is in the expected format
        assert (
            len(input_image_torch.shape) == 3
            and input_image_torch.shape[2] == 3
            and input_image_torch.shape[0] == self.model.inp_size
            and input_image_torch.shape[1] == self.model.inp_size
        ), f"Input image must be HWC with 1024x1024x3, not {input_image_torch.shape}."

        # Change the order of dimensions to BCHW
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        self.input_image_torch = input_image_torch
        self.features = self.model.image_encoder(self.input_image_torch)

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True):
        if self.input_image_torch is None:
            raise ValueError("Please set the input image first.")
        
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            assert (
                len(point_coords.shape) == 2 and point_coords.shape[1] == 2
            ), "point_coords must be a Nx2 array of (X,Y) coordinates."
            assert (
                len(point_labels.shape) == 1 and point_labels.shape[0] == point_coords.shape[0]
            ), "point_labels must be a length N array corresponding to point_coords."
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            
        if box is not None:
            assert (
                len(box.shape) == 1 and box.shape[0] == 4
            ), "box must be a length 4 array in XYXY format."
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
            
        if mask_input is not None:
            assert (
                len(mask_input.shape) == 3 and mask_input.shape[0] == 1 and mask_input.shape[1] == mask_input.shape[2] == 256
            ), "mask_input must be a 1xHxW array, where H=W=256."
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()

        del masks, iou_predictions, coords_torch, labels_torch, box_torch, mask_input_torch, point_coords, point_labels, box, mask_input
        torch.cuda.empty_cache()
        return masks_np, iou_predictions_np, 

    def predict_torch(self, point_coords: Optional[torch.Tensor], point_labels: Optional[torch.Tensor], boxes: Optional[torch.Tensor] = None, mask_input: Optional[torch.Tensor] = None, multimask_output: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_image_torch is None:
            raise ValueError("Please set the input image first.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        if hasattr(self.model, "mask_generator"):
            if mask_input is not None:
                raise ValueError("Model has mask generator, do not accept mask_input.")
            else:
                mask_input = self.model.mask_generator(self.input_image_torch)

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.model.inp_size, self.model.inp_size)
        masks = masks > self.mask_threshold
        del sparse_embeddings, dense_embeddings, low_res_masks, point_coords, point_labels, boxes, mask_input
        torch.cuda.empty_cache()
        return masks, iou_predictions
    
    @property
    def device(self) -> torch.device:
        return self.model.device

def config_modify(config):
    if config.get('PYTORCH_CUDA_ALLOC_CONF') is not None:
        max_split_size_mb = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        print(f"PYTORCH_CUDA_ALLOC_CONF:{max_split_size_mb}")
        config['PYTORCH_CUDA_ALLOC_CONF'] = max_split_size_mb
    
    for dataset in ['train_dataset', 'val_dataset', 'test_dataset']:
        config[dataset]['dataset']['args']['root_path_1'] = config[dataset]['dataset']['args']['root_path_1'].replace('datasetname', config['dataset_name'])
        config[dataset]['dataset']['args']['root_path_2'] = config[dataset]['dataset']['args']['root_path_2'].replace('datasetname', config['dataset_name'])
    
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    # instantiate the dataset of paired-image-folders of train and val
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    # instantiate the dataset of wrapper of train and val
    print('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if k != 'filename':
            # k,v = 'inp', self.img_transform(img); 'gt', self.mask_transform(mask)
            print('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=16, pin_memory=True)
    return loader

def make_data_loaders():
    test_loader = make_data_loader(config.get('test_dataset'), tag='test')
    return test_loader

def prepare_testing():
    model = models.make(config['model']).cuda()
    model = torch.nn.DataParallel(model).cuda()
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)

    model.eval()
  
    return model

def main(config, save_path):
    config = config_modify(config)
    test_loader = make_data_loaders()
    model = prepare_testing()

    # Instantiate predictor
    predictor = Predictor(model)

    # Iterate over the dataset
    for idx, data in enumerate(tqdm(test_loader)):
        input_data = data["image"]
        # Input image must be HWC with 1024x1024x3, not torch.Size([1, 3, 1024, 1024])
        input_data = input_data.squeeze(0).permute(1, 2, 0).numpy()

        # Predict
        predictor.set_image(input_data)
        masks_np, iou_predictions_np = predictor.predict(multimask_output=False)

        # Save the output
        output_img = Image.fromarray((masks_np[0] * 255).astype(np.uint8))
        output_img.save(os.path.join(save_path, f"prediction_{data['filename'][0]}.png"))
        torch.cuda.empty_cache()

    print("Prediction completed and saved to the disk.")

if __name__ == '__main__':
    device = torch.device("cuda:0")

    # Load the config file
    config_path = "configs/iou-sam-vit-b_test.yaml"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    
    save_prefix = str(config.get('sam_prefix'))
    save_postfix = str(config.get('modeltype'))
    save_name = save_prefix + '-' + config.get('dataset_name') + '_' + save_postfix
    # Prepare saving paths
    save_path = os.path.join('./save', save_name, "prediction_output")
    utils.ensure_path(save_path)
    print('save path : {} is cleared.'.format(save_path))

    main(config, save_path)
