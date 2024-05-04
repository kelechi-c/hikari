import torch
import torchvision
import torch.nn as nn


device = ('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(device=device)


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
        
            
    
class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    
    def forward(self, x):
        return x
    
    
