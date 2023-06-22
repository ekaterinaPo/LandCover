import logging
import boto3
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from UNet_model import UNet
import os
from io import BytesIO
#import base64
from mangum import Mangum
#import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

logger = logging.getLogger()
logger.setLevel(logging.INFO)


bucket_name = 'mlops-deploy'
model_prefix = 'LandCover/model/'



def reverse_one_hot(mask_img, label_rgb_values, image_size):
    mask_img_reconstructed = np.zeros((image_size, image_size, 3))
    mask_img = np.argmax(mask_img, axis = 0)
    for idx, colour in enumerate(label_rgb_values):
        mask_img_reconstructed[np.equal(mask_img, idx)] = colour
    return mask_img_reconstructed

def get_most_recent_s3_object(bucket, prefix):
    s3_client = boto3.client('s3', config=boto3.session.Config(
        region_name='us-east-1'
    ))
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
    s3_client = boto3.client('s3', config=boto3.session.Config(
        region_name='us-east-1'
    ))
    logger.info('Loading model from: s3://%s/%s', bucket, prefix)
    model = UNet(in_channels=3, out_channels=7, features=64)

    latest = get_most_recent_s3_object(bucket, prefix)
    
    with BytesIO() as f:
        s3_client.download_fileobj(Bucket = bucket, Key = latest["Key"], Fileobj = f)
        f.seek(0)
        model_weights = torch.load(f)

    model.load_state_dict(model_weights)
    logger.info('Model loaded successfully')
    return model

def predict(image, base_filename, image_size):
    buf_image = BytesIO()
    image.save(buf_image, format="PNG")
    byte_im = buf_image.getvalue()

    s3_client = boto3.client('s3', config=boto3.session.Config(
        region_name='us-east-1'
    ))
    s3_image_key = f'{base_filename}.jpg'
    s3_client.put_object(
        Bucket='landcover-prediction',
        Key=s3_image_key,
        Body=byte_im,
        ContentType="image/jpg"
    )
    img_resized = Image.open(BytesIO(byte_im)).resize((image_size, image_size), resample=Image.BILINEAR)
    img_np = np.array(img_resized)
    img_tensor = transforms.ToTensor()(img_np)
           
    #model = load_model(model_path = "C:/Users/kate/Desktop/WeCloudData/project/LandCover/UNET_20230521-1753_1024_full_epoch_1.pth")
    model = load_model(bucket = bucket_name, prefix = model_prefix)
    #get prediction
    prediction = model(img_tensor.unsqueeze(dim = 0))
    #reverse to mask
    label_rgb_values = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
    reverse_one_hot_encoded_mask = reverse_one_hot(prediction[0].detach().numpy(), label_rgb_values, image_size = image_size).astype(int)

    # Save the result as a PNG image
    result_image = Image.fromarray(reverse_one_hot_encoded_mask.astype(np.uint8))
    buf = BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    modified_filename = base_filename.replace("_sat", "_mask")
    result_prefix = f'LandCover/results/{modified_filename}.png'
    
    s3_client.put_object(
        Bucket="mlops-deploy",
        Key=result_prefix,
        Body=byte_im,
        ContentType="image/png"
    )

    return result_image 

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

IMAGE_DIR = 'images/'

app = FastAPI()
handler=Mangum(app)

@app.get("/")
def read_root():
    return {"model": "Semantic Segmentation"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    base_filename = os.path.splitext(file.filename)[0]

    if not extension:
        return "Image must be jpg or png format!"
    
    contents = await file.read()
    image = read_imagefile(contents)
    image_size = 512
    result_image = predict(image, base_filename, image_size)
    
    modified_maskname = base_filename.replace("_sat", "_mask")
    result_filename = f'{modified_maskname}_result.png'
    result_path = os.path.join(IMAGE_DIR, result_filename)
    result_image.save(result_path, format="PNG")

    return FileResponse(result_path)
