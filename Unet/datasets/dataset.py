from PIL import Image
import os
from torchvision import transforms
import torch
import random

class JointTransform: 
    def __init__(self, input_size):
        self.transform = transforms.Compose([
            transforms.RandomAffine(20, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

    def __call__(self, image, mask):
        rand_seed = random.randint(0, 2**32)  # Generate a new seed for every call
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)  # Ensure image and mask get the same transformation
        image = self.transform(image)
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
        mask = self.transform(mask)

        torch.cuda.empty_cache()
        return image, mask


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, input_size=1024, need_transform=False, ratio=1.0):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.need_transform = need_transform
        self.input_size = input_size
        self.joint_transform = JointTransform(input_size)
        self.image_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])

        self.filepaths = self.get_filepaths(ratio)
        random.shuffle(self.filepaths)  # Finally, shuffle the whole dataset

    def get_filepaths(self, ratio):
        filepaths = []
        expected_size = 0
        folders = [f for f in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir, f)) and f[-1].isdigit()]
        random.shuffle(folders)  # Shuffle the folders

        for folder in folders:
            images = [f for f in os.listdir(os.path.join(self.img_dir, folder)) if "image" in f]
            expected_size += len(images)
        
        expected_size = int(expected_size * ratio)

        for folder in folders:
            images = [f for f in os.listdir(os.path.join(self.img_dir, folder)) if "image" in f]
            random.shuffle(images)  # Shuffle the images within each folder
            images = images[:int(len(images) * ratio)]
            for image in images:
                filepaths.append(os.path.join(folder, image))
        
        while len(filepaths) < expected_size:
            # Fetch more images from the shuffled folders
            for folder in folders:
                images = [f for f in os.listdir(os.path.join(self.img_dir, folder)) if "image" in f]
                additional_images = images[:expected_size - len(filepaths)]
                for image in additional_images:
                    filepaths.append(os.path.join(folder, image))
        
        # If we have more images than expected_size, truncate the list.
        if len(filepaths) > expected_size:
            filepaths = filepaths[:expected_size]
        
        return filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx, require_folder=False):
        img_path = os.path.join(self.img_dir, self.filepaths[idx])
        gt_path = os.path.join(self.gt_dir, self.filepaths[idx].rsplit('.', 1)[0].replace("image", "label") + '.png')
        folder = self.filepaths[idx].split('/')[0]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(gt_path).convert('L')

        if self.need_transform:
            image, mask = self.joint_transform(image, mask)
        else:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        torch.cuda.empty_cache()
        
        if require_folder:
            return image, mask, folder
        else:
            return image, mask


