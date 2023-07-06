import streamlit as st
from PIL import Image
import requests
import base64
import json
import numpy as np

st.set_page_config(layout="wide", page_title="LandCover Semantic Segmentation")

st.write("## Create Segmentation mask")
st.write(":dog: Try uploading an image.")
st.sidebar.write("## Upload and download :gear:")

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    # Convert uploaded image to base64
    img = Image.open(my_upload)
    img_data = img.tobytes()
    img_base64 = base64.b64encode(img_data).decode('utf-8')

    # Prepare API request data
    api_url = 'https://dj0c126i93.execute-api.us-east-1.amazonaws.com/test/segmentation'
    payload = {
        'image': img_base64
    }

    # Send API request
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        result = json.loads(response.text)
        encoded_image = result['encoded_image']
        result_image = Image.open(BytesIO(base64.b64decode(encoded_image)))
        col1.image(img)
        col2.image(np.fliplr(np.rot90(result_image, 3)))
    else:
        st.write(f"Error: {response.status_code}")
