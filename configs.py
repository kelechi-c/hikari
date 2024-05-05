import torch
import cv2

class Config:
    epochs = 35
    batch_size = 64
    num_classes = 1
    num_channels = 1
    num_levels = 3
    lr = 0.001
    height = 512
    width = 512
    base_dir = '/kaggle/input/'
    image_data_path = ""
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model_output_path = 'hikari_image_segment.pth'
    momentum = 0.9


def read_img(image_file, config):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.height, config.width))

    return image

