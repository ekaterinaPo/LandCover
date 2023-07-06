import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)
    
class DownStep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownStep, self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            DoubleConv(in_channels, out_channels),
            )
        
    def forward(self, x):
        return self.maxpool(x)
    
class UpStep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpStep, self).__init__()
        self.up_step = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        #self.double_conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.up_step(x)
        #x = torch.cat([x, concat], dim=1)
        return x
    
class UNet(nn.Module):  
    def __init__(self, in_channels = 3, out_channels = 7, features = 64):
        super(UNet, self).__init__()
        #Encoder
        self.down_1 = DoubleConv(in_channels, features)
        self.down_2 = DownStep(features, features*2)
        self.down_3 = DownStep(features*2, features*4)
        self.down_4 = DownStep(features*4, features*8)
        #Bottleneck
        self.centre =  DownStep(features*8, features*16)
        #Decoder
        self.up4 = UpStep(features * 16, features * 8)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.up3 = UpStep(features * 8, features * 4)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.up2 = UpStep(features * 4, features * 2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.up1 = UpStep(features * 2, features)
        self.decoder1 = DoubleConv(features * 2, features)

        self.out = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        #Encoder
        enc1 = self.down_1(x)
        enc2 = self.down_2(enc1)
        enc3 = self.down_3(enc2)
        enc4 = self.down_4(enc3)
        
        #Bottleneck
        bottleneck = self.centre(enc4)

        #Decoder
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1) # dim=1
        dec4 = self.decoder4(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1) # dim=1
        dec3 = self.decoder3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1) # dim=1
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1) # dim=1
        dec1 = self.decoder1(dec1)
        out = self.out(dec1)
        
        out = torch.nn.functional.softmax(out, dim = 1)
        
        return out

