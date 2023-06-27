import logging
import boto3
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from UNet_model import UNet
import os
from io import BytesIO
from mangum import Mangum
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import base64
import json
import torchvision.transforms as transforms

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS S3 client
s3_client = boto3.client('s3', region_name='us-east-1')

bucket_name = 'mlops-deploy'
model_prefix = 'LandCover/model/'
image_dir = '/tmp'

def reverse_one_hot(mask_img, label_rgb_values, image_size):
    mask_img_reconstructed = np.zeros((image_size, image_size, 3))
    mask_img = np.argmax(mask_img, axis=0)
    for idx, colour in enumerate(label_rgb_values):
        mask_img_reconstructed[np.equal(mask_img, idx)] = colour
    return mask_img_reconstructed

def get_most_recent_s3_object(bucket, prefix):
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    latest = None
    for page in page_iterator:
        if "Contents" in page:
            latest2 = max(page['Contents'], key=lambda x: x['LastModified'])
            if latest is None or latest2['LastModified'] > latest['LastModified']:
                latest = latest2
    return latest

def load_model(bucket=bucket_name, prefix=model_prefix):
    logger.info('Loading model from: s3://%s/%s', bucket, prefix)
    model = UNet(in_channels=3, out_channels=7, features=64)

    latest = get_most_recent_s3_object(bucket, prefix)
    
    with BytesIO() as f:
        s3_client.download_fileobj(Bucket=bucket, Key=latest["Key"], Fileobj=f)
        f.seek(0)
        model_weights = torch.load(f)

    model.load_state_dict(model_weights)
    logger.info('Model loaded successfully')
    return model

def predict(image, base_filename, image_size):
    img_resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    img_np = np.array(img_resized)
    img_tensor = transforms.ToTensor()(img_np)
           
    model = load_model(bucket=bucket_name, prefix=model_prefix)
    
    # Get prediction
    prediction = model(img_tensor.unsqueeze(dim=0))
    
    # Reverse to mask
    label_rgb_values = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
    reverse_one_hot_encoded_mask = reverse_one_hot(prediction[0].detach().numpy(), label_rgb_values, image_size=image_size).astype(int)

    # Save the result as a PNG image
    result_image = Image.fromarray(reverse_one_hot_encoded_mask.astype(np.uint8))
    buf = BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    modified_filename = base_filename.replace("_sat", "_mask")
    result_prefix = f'LandCover/results/{modified_filename}.png'
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=result_prefix,
        Body=byte_im,
        ContentType="image/png"
    )

    return result_image 

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


app = FastAPI()
handler = Mangum(app)


@app.get("/")
def read_root():
    return {"model": "Semantic Segmentation"}


@app.post('/json-image')
async def dictionaary_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    base_filename = os.path.splitext(file.filename)[0]

    if not extension:
        return "Image must be jpg or png format!"
    
    contents = await file.read()
    image = read_imagefile(contents)
    image_size = 512
    result_image = predict(image, base_filename, image_size)
    
    modified_maskname = base_filename.replace("_sat", "_mask")
    result_filename = f'{modified_maskname}.png'
    result_path = os.path.join(image_dir, result_filename)
    result_image.save(result_path, format="PNG")
    
    # Upload result image to S3
    prefix_request = 'LandCover/requests'

    s3_client.upload_file(result_path, bucket_name, 
                          os.path.join(prefix_request, result_filename))
    
    # Convert the result image to a JSON-serializable format
    result_array = np.array(result_image)
    prediction_json = json.dumps(result_array.tolist())

    return {"prediction": prediction_json}
    
@app.post("/predict-s3-link")
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
    result_filename = f'{modified_maskname}.png'
    
    # Convert result image to bytes
    with BytesIO() as f:
        result_image.save(f, format="PNG")
        byte_im = f.getvalue()
    
    # Upload result image to S3
    result_prefix = f'LandCover/results/{result_filename}'
    s3_client.put_object(
        Bucket=bucket_name,
        Key=result_prefix,
        Body=byte_im,
        ContentType="image/png"
    )
    
    # Return the S3 URL of the result image
    result_url = f"s3://{bucket_name}/{result_prefix}"
    return {"result_url": result_url}


     

