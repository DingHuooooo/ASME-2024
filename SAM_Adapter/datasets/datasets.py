from PIL import Image
import os
from torchvision import transforms
import torch
import random

class JointTransform: 
    def __init__(self, input_size):
        self.transform = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),  # Move ToTensor to the end
        ])
        #self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        self.rand_seed = random.randint(0, 2**32)

    def __call__(self, image, mask):
        torch.manual_seed(self.rand_seed)
        random.seed(self.rand_seed)  # Ensure image and mask get the same transformation
        image = self.transform(image)
        #image = self.normalize(image)
        torch.manual_seed(self.rand_seed)
        random.seed(self.rand_seed)
        mask = self.transform(mask)

        torch.cuda.empty_cache()
        return image, mask


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, input_size=1024, need_transform=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.need_transform = need_transform
        self.input_size = input_size
        self.filenames = [f for f in os.listdir(img_dir) if 'image' in f]

        self.joint_transform = JointTransform(input_size)
        self.single_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),  # Move ToTensor to the end
            #transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.filenames[idx])
        gt_name = os.path.join(self.gt_dir, self.filenames[idx].rsplit('.', 1)[0].replace("image", "label") + '.png')

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(gt_name).convert('L')
        
        if self.need_transform:
            image, mask = self.joint_transform(image, mask)
        else:
            image = self.single_transform(image)
            mask = transforms.ToTensor()(transforms.Resize((self.input_size, self.input_size))(mask))

        torch.cuda.empty_cache()
        
        return image, mask
