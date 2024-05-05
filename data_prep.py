import cv2
import torchvision
from torch import tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from configs import Config

data_folder_path = Config.image_data_path
height = Config.height
width = Config.width

class ImageDataset(Dataset):
    def __init__(self, image_paths: str, mask_paths: str, transforms):
        super().__init__()
        self.transforms = transforms
        self.images = image_paths
        self.masks = mask_paths
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.mask[idx]
        
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height, width))
        
        mask = cv2.imread(self.masks[idx], 0)
        
        image = self.transforms(image)
        mask = self.transforms(mask)
        
        return (image, mask)

images = sorted()
masks = sorted()

image_transforms = torchvision.transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((Config.height, Config.width)),
    transforms.ToTensor()
])

image_mask_data = ImageDataset(image_paths=images, mask_paths=masks, transforms=image_transforms)

train_size = 0.8 * len(image_mask_data)
valid_size = len(image_mask_data) - train_size

train_data, valid_data = random_split(image_mask_data, [train_size, valid_size])

train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=Config.batch_size, shuffle=True)