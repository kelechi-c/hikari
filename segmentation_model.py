from networkx import is_planar
import torch
import torchvision
from torch import nn, relu

# 221029255650


device = ('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(device=device)

class encoder_conv_block(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output, output, kernel_size=3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_block(x)

        return x


class Unet_model(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # Encoder
        # self.encoder = Encoder()
        self.ec1 = encoder_conv_block(3, 64)
        self.ec2 = encoder_conv_block(64, 128)
        self.ec3 = encoder_conv_block(128, 256)
        self.ec4 = encoder_conv_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encode_step_without_pool = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        # self.decoder = Decoder()

        self.conv_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        xe1 = self.ec1(x)
        xe2 = self.ec2(xe1)
        xe3 = self.ec4(xe2)
        xe4 = self.ec3(xe3)

        xe5 = self.encode_step_without_pool(xe4)

        xd1 = self.conv_transpose1(xe5)
        xd1 = torch.cat([xd1, xe4], dim=1)
        xd1 = self.dec1(xd1)

        xd2 = self.conv_transpose2(xd1)
        xd2 = torch.cat([xd2, xe3], dim=1)
        xd2 = self.dec2(xd2)

        xd3 = self.conv_transpose3(xd2)
        xd3 = torch.cat([xd3, xe2], dim=1)
        xd3 = self.dec3(xd3)

        xd4 = self.conv_transpose4(xd3)
        xd4 = torch.cat([xd4,xe1], dim=1)
        xd4 = self.dec4(xd4)
        
        output = self.out_conv(xd4)

        return output


hikari_segmenter_model = Unet_model().to(device)
