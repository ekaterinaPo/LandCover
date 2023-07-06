import json
import base64
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from UNet_model import UNet

def predict_image(upload_image):
    bucket_name = 'mlops-deploy'
    model_prefix = 'LandCover/model/'
    image_size = 512
    
    img_resized = upload_image.resize((image_size, image_size), resample=Image.BILINEAR)
    img_np = np.array(img_resized)
    img_tensor = transforms.ToTensor()(img_np)

    model = load_model(bucket_name, model_prefix)

    # Get prediction
    prediction = model(img_tensor.unsqueeze(dim=0))

    # Reverse to mask
    label_rgb_values = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
    reverse_one_hot_encoded_mask = reverse_one_hot(prediction[0].detach().numpy(), label_rgb_values,
                                                  image_size=image_size).astype(int)

    # Save the result as a PNG image
    result_image = Image.fromarray(reverse_one_hot_encoded_mask.astype(np.uint8))
    
    # Convert the result image to base64 encoding
    buffered = BytesIO()
    result_image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return encoded_image

def load_model(bucket, prefix):
    s3_client = boto3.client("s3")
    model = UNet(in_channels=3, out_channels=7, features=64)
    latest_model = get_most_recent_s3_object(bucket, prefix)
    with BytesIO() as f:
        s3_client.download_fileobj(Bucket=bucket, Key=latest_model["Key"], Fileobj=f)
        f.seek(0)
        model_weights = torch.load(f)
    model.load_state_dict(model_weights)
    return model

def reverse_one_hot(mask_img, label_rgb_values, image_size):
    mask_img_reconstructed = np.zeros((image_size, image_size, 3))
    mask_img = np.argmax(mask_img, axis=0)
    for idx, colour in enumerate(label_rgb_values):
        mask_img_reconstructed[np.equal(mask_img, idx)] = colour
    return mask_img_reconstructed

def get_most_recent_s3_object(bucket, prefix):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    latest = None
    for page in page_iterator:
        if "Contents" in page:
            latest2 = max(page['Contents'], key=lambda x: x['LastModified'])
            if latest is None or latest2['LastModified'] > latest['LastModified']:
                latest = latest2
    return latest


def lambda_handler(event, context):
    image_base64 = event['image']
    image_data = base64.b64decode(image_base64)
    upload_image = Image.open(BytesIO(image_data))
    encoded_image = predict_image(upload_image)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'encoded_image': encoded_image}),
        'headers': {
            'Content-Type': 'application/json'
        }
    }