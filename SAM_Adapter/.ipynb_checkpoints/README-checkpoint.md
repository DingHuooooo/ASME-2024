1. When preprocessing data, if the data varies a lot from ImageNet dataset, maybe need to change the mean and std transforms.Normalize.
    
    # Load your training data
    # If your data doesn't fit into memory you need to load it in chunks and update the mean and std accordingly
    data = ...

    # Convert data to torch.Tensor
    data = torch.Tensor(data)

    # Per channel mean and standard deviation
    mean = torch.mean(data, dim=[0,2,3])
    std = torch.std(data, dim=[0,2,3])


2. first_k can be used in config to only load first k images.

3. In class PairedImageFolders(Dataset), self.dataset_1 = ImageFolder(root_path_1, **kwargs), maybe we can add mask=True to load image as Grayscale image.

4. In class TrainDataset(Dataset), maybe gt_resize can be removed.

5. We can do resize in ImageFolder and TrainDataset and ValDataset.

6. The datatype of the tensors also matters. For example, if the image is represented as floats (torch.float32) and the mask as integers (torch.int), the image might take up more memory even if the number of elements is the same.

7. loss mode can be changed to iou by directly modifying self.loss_mode in class SAM(nn.Module)

8. by default multimask_output=False in class SAM(nn.Module) under def forward()

9. iou_predictions maybe can be output in class SAM(nn.Module) under def forward()

10. the features in Adaptor can be generaterd in different ways, and then being forward and added to x. difference with VPT?

11. If adjust to Multi-Class Classification Problem, calc for metrics must be changed

12. in sam.py multimask_output can control multiclass/binary class

13. calculate IOU before/after binarization/threshlod

14. output of model (after MLP) should be fed into sigmoid(), then calculate loss, or generate masks(without min-max scaling)








## SAM Fails to Segment Anything?—SAM-adapter: Adapting SAM in Underperformed Scenes

Tianrun Chen, Lanyun Zhu, Chaotao Ding, Runlong Cao, Yan Wang, Zejian Li, Lingyun Sun, Papa Mao, Ying Zang

<a href='https://www.kokoni3d.com/'> KOKONI, Moxin Technology (Huzhou) Co., LTD </a>, Zhejiang University, Singapore University of Technology and Design, Huzhou University, Beihang University.

  <a href='https://tianrun-chen.github.io/SAM-Adaptor/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
## 

<a href='https://arxiv.org/abs/2304.09148'><img src='https://img.shields.io/badge/ArXiv-2304.09148-red' /></a> 

Update on 28 April: We tested the performance of polyp segmentation to show our approach can also work on medical datasets.
<img src='https://tianrun-chen.github.io/SAM-Adaptor/static/images/polyp.jpg'>
Update on 22 April: We report our SOTA result based on ViT-H version of SAM (use demo.yaml). We have also uploaded the yaml config for ViT-L and ViT-B version of SAM, suitable  GPU with smaller memory (e.g. NVIDIA Tesla V-100), although they may compromise on accuracy.

## Environment
This code was implemented with Python 3.8 and PyTorch 1.13.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```


## Quick Start
1. Download the dataset and put it in ./load.
2. Download the pre-trained [SAM(Segment Anything)](https://github.com/facebookresearch/segment-anything) and put it in ./pretrained.
3. Training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 loadddptrain.py --config configs/demo.yaml
```
!Please note that the SAM model consume much memory. We use 4 x A100 graphics card for training. If you encounter the memory issue, please try to use graphics cards with larger memory!


4. Evaluation:
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train.py --nnodes 1 --nproc_per_node 4 --config [CONFIG_PATH]
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Pre-trained Models
To be uploaded

## Dataset

### Camouflaged Object Detection
- **[COD10K](https://github.com/DengPingFan/SINet/)**
- **[CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)**
- **[CHAMELEON](https://www.polsl.pl/rau6/datasets/)**

### Shadow Detection
- **[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)**

### Polyp Segmentation - Medical Applications
- **[Kvasir](https://datasets.simula.no/kvasir-seg/)**

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{chen2023sam,
      title={SAM Fails to Segment Anything? -- SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, and More}, 
      author={Tianrun Chen and Lanyun Zhu and Chaotao Ding and Runlong Cao and Shangzhan Zhang and Yan Wang and Zejian Li and Lingyun Sun and Papa Mao and Ying Zang},
      year={2023},
      eprint={2304.09148},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
The part of the code is derived from Explicit Visual Prompt   <a href='https://nifangbaage.github.io/Explicit-Visual-Prompt/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> by 
Weihuang Liu, [Xi Shen](https://xishen0220.github.io/), [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/), and [Xiaodong Cun](https://vinthony.github.io/) by University of Macau and Tencent AI Lab.

