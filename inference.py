import torch
import torchvision
import cv2 
from PIL import Image
from configs import Config, read_img

height = Config.height
width = Config.width

image_model_path = Config.model_output_path

