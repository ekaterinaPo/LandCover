import logging
import json
import boto3
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from UNet_model import UNet
import io
import os
from io import BytesIO


logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

bucket_name = 'mlops-deploy'
model_prefix = 'LandCover/model/'

PREDICT_PATH = 'LandCover/results/'



def reverse_one_hot(mask_img, label_rgb_values, image_size):
    mask_img_reconstructed = np.zeros((image_size, image_size, 3))
    mask_img = np.argmax(mask_img, axis = 0)
    for idx, colour in enumerate(label_rgb_values):
        mask_img_reconstructed[np.equal(mask_img, idx)] = colour
    return mask_img_reconstructed

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
    logger.info('Loading model from: s3://%s/%s', bucket, prefix)
    model = UNet(in_channels=3, out_channels=7, features=64)

    latest = get_most_recent_s3_object(bucket, prefix)
    #latest_model = s3_client.get_object(Bucket = bucket, Key = latest["Key"])

    #obj = s3_res.Bucket(bucket_name).Object(latest["Key"])
    with BytesIO() as f:
        s3_client.download_fileobj(Bucket = bucket, Key = latest["Key"], Fileobj = f)
        f.seek(0)
        model_weights = torch.load(f)

    model.load_state_dict(model_weights)
    logger.info('Model loaded successfully')
    return model

def predict(upload_image, object_key, image_size):
    img_resized = upload_image.resize((image_size, image_size), resample=Image.BILINEAR)
    img_np = np.array(img_resized)
    img_tensor = transforms.ToTensor()(img_np)
    
    bucket_name = "mlops-deploy"
    
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
    
    # Save the byte data to a file
    #file_path = 'C:/Users/kate/Desktop/WeCloudData/project/LandCover/result.png'
    #with open(file_path, "wb") as file:
    #    file.write(byte_im)
        
    # Save the byte data to the S3 bucket
    key_name = os.path.splitext(object_key)[0]
    result_prefix = f'LandCover/results/{key_name}.png'
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket=bucket_name,
        Key=result_prefix,
        Body=byte_im,
        ContentType="image/png"
    )

    return result_image #reverse_one_hot_encoded_mask

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    # Retrieve the uploaded image from the S3 event
    bucket_trigger = event['Records'][0]['s3']['bucket']['name']
    object_key = event['Records'][0]['s3']['object']['key']

    # Extract image data from the API Gateway event
    #body = json.loads(event['body'])
    #image_data = body['image']
    # Retrieve the uploaded image information
    #bucket_trigger = body['bucket']
    #object_key = body['objectKey']

    print(bucket_trigger)
    print (object_key)

    # Download the uploaded image from S3
    response = s3.get_object(Bucket=bucket_trigger, Key=object_key)
    image_data = response['Body'].read()

    # Create an image object from the downloaded data
    upload_image = Image.open(BytesIO(image_data))
    print("UPLOAD IMAGE - DONE")
    
    # Generate the prediction
    image_size = 512
    result_image = predict(upload_image, object_key, image_size)

    # Save the result as a PNG image
    #result_image = Image.fromarray(result.astype(np.uint8))
    #result_image.save(file_path, format="PNG")

    # Return a response indicating the successful execution
    return {
        'statusCode': 200,
        'body': json.dumps('Prediction completed successfully.')
    }
