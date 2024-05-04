import torch
from torch.utils.data import Dataset, DataLoader, random_split

data_folder_path = ''

class ImageDataset(Dataset):
    def __init__(self, image_folder):
        super().__init__()
        self.image_folder = image_folder
        
    def __len__(self):
        return len()
    
    def __getitem__(self, idx):
        
        return 
    

image_mask_data = ImageDataset(data_folder_path)

train_size = 0.8 * len(image_mask_data)
valid_size = len(image_mask_data) - train_size

train_data, valid_data = random_split(image_mask_data, [train_size, valid_size])