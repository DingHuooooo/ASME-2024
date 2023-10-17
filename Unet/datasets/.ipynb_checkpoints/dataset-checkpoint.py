from PIL import Image
import os
from torchvision import transforms
import torch
import random

from PIL import Image
import os
from torchvision import transforms
import torch
import random

class JointTransform:
    def __init__(self, input_size):
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
        ])
        self.to_tensor = transforms.ToTensor()
        self.rand_seed = random.randint(0, 2**32)

    def __call__(self, image, mask):
        torch.manual_seed(self.rand_seed)
        random.seed(self.rand_seed)  # Ensure image and mask get the same transformation
        image = self.transform(image)
        torch.manual_seed(self.rand_seed)
        random.seed(self.rand_seed)
        mask = self.transform(mask)

        return self.to_tensor(image), self.to_tensor(mask)


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
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.filenames[idx])
        gt_name = os.path.join(self.gt_dir, self.filenames[idx].rsplit('.', 1)[0].replace("image", "label") + '.png')

        image = Image.open(img_name).convert('RGB')
        gray_image = image.convert("L")
        image = Image.merge("RGBA", (image.split() + (gray_image,)))
        mask = Image.open(gt_name).convert('L')
        
        if self.need_transform:
            image, mask = self.joint_transform(image, mask)
        else:
            image = self.single_transform(image)
            mask = self.single_transform(mask)

        del gray_image
        torch.cuda.empty_cache()
        
        return image, mask


