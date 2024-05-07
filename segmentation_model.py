import torch
import torchvision
import torch.nn as nn

# 221029255650


device = ('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(device=device)

class encoder_conv_block(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.Conv2d(output, output, kernel_size=3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_block(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encode_step1 = nn.Sequential(
        #     encoder_conv_block(3, 64),
        #     encoder_conv_block(64, 128),
        #     encoder_conv_block(128, 256),
        #     encoder_conv_block(256, 512),
        # )
        self.ec1 = encoder_conv_block(3, 64)
        self.ec2 = encoder_conv_block(64, 128)
        self.ec3 = encoder_conv_block(128, 256)
        self.ec4 = encoder_conv_block(256, 512)

        self.encode_step_without_pool = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.ec1(x)
        x = self.ec2(x)
        x = self.ec3(x)
        x = self.ec4(x)
        x = self.encode_step_without_pool(x)
        x = nn.ReLU(x)

        return x


# class Decoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.dec1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
#         self.dec1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
#         self.conv_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.d2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.d2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

#         self.conv_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.d3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
#         self.d3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

#         self.conv_transpose4= nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.d4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.d4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

#     def forward(self, x):
        
#         return x


class Unet_Hikari(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Encoder
        # self.encoder = Encoder()
        self.ec1 = encoder_conv_block(3, 64)
        self.ec2 = encoder_conv_block(64, 128)
        self.ec3 = encoder_conv_block(128, 256)
        self.ec4 = encoder_conv_block(256, 512)

        self.encode_step_without_pool = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        )

        # Decoder
        # self.decoder = Decoder()
        self.conv_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        
        x1 = self.ec2(self.ec1(x))
        x2 = self.ec4(self.ec3(x1))
        xe3 = self.encode_step_without_pool(x2)
        
        
        

        return x


hikari_segmenter = Unet_Hikari(n_classes=10).to(device)
