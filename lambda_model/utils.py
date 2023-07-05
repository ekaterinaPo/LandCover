import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch
import boto3
from io import BytesIO

#s3_client = boto3.client('s3')

bucket_name = 'mlops-deploy'
model_prefix = 'LandCover/model/'

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
        return x
    
class UNet(nn.Module):  
    def __init__(self, in_channels = 3, out_channels = 7, features = 64):
        super(UNet, self).__init__()
        #Encoder
        self.down_1 = DoubleConv(in_channels, features)
        self.down_2 = DownStep(features, features*2)
        self.down_3 = DownStep(features*2, features*4)
        self.down_4 = DownStep(features*4, features*8)
    
        self.centre =  DownStep(features*8, features*16)
  
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

def get_most_recent_s3_object(bucket, prefix):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator( "list_objects_v2" )
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    latest = None
    for page in page_iterator:
        if "Contents" in page:
            latest2 = max(page['Contents'], key=lambda x: x['LastModified'])
            if latest is None or latest2['LastModified'] > latest['LastModified']:
                latest = latest2
    return latest

def load_model(bucket = bucket_name, prefix = model_prefix):
    #s3_res = boto3.resource('s3')
    s3_client = boto3.client("s3")
    model = UNet(in_channels=3, out_channels=7, features=64)

    latest = get_most_recent_s3_object(bucket, prefix)
    #latest_model = s3_client.get_object(Bucket = bucket, Key = latest["Key"])

    #obj = s3_res.Bucket(bucket_name).Object(latest["Key"])
    with BytesIO() as f:
        s3_client.download_fileobj(Bucket = bucket, Key = latest["Key"], Fileobj = f)
        f.seek(0)
        model_weights = torch.load(f)

    model.load_state_dict(model_weights)
    return model



#def load_model(model_path: str):
#    assert not model_path.startswith("s3://"), "Loading from S3 is not currently supported"
#    model = UNet(in_channels = 3, out_channels = 7, features = 64)
#    model.load_state_dict(torch.load(model_path))
#    return model

#model = load_model(model_path = "C:/Users/kate/Desktop/WeCloudData/project/LandCover/UNET_20230521-1753_1024_full_epoch_1.pth")

def reverse_one_hot_map_cg(one_hot_map, label_rgb_values):
    """
    Convert one-hot encoded mask to an RGB image with shape (H, W, 3)
    using the provided label RGB values.
    """
    class_indices = np.argmax(one_hot_map, axis=-1)
    rgb_image = np.zeros_like(one_hot_map[..., 0:3])  # Assuming RGB is the first three channels
    for i, rgb_value in enumerate(label_rgb_values):
        mask = (class_indices == i)
        rgb_image[mask] = rgb_value
    return rgb_image


def prediction(image, image_size = 512):
    #transform the image
    img_resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    img_np = np.array(img_resized)
    img_tensor = transforms.ToTensor()(img_np)
    model = load_model(bucket = bucket_name, prefix = model_prefix)
    #get prediction
    prediction = model(img_tensor.unsqueeze(dim = 0))
    
    #reverse to mask
    label_rgb_values = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
    reverse_one_hot_encoded_mask = reverse_one_hot_map_cg(prediction[0].detach().numpy().transpose(2, 1, 0), label_rgb_values)
    #reverse_one_hot_encoded_mask = reverse_one_hot(prediction[0].detach().numpy(), label_rgb_values, image_size = image_size).astype(int)
    return reverse_one_hot_encoded_mask/255