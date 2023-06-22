import streamlit as st
import torch
from PIL import Image
from io import BytesIO
import base64
from torchvision import transforms
import numpy as np
from utils import prediction


st.set_page_config(layout="wide", page_title="LandCover Semantic Segmentation")

st.write("## Create Segmentation mask")
st.write(
    ":dog: Try uploading an image."
)
st.sidebar.write("## Upload and download :gear:")



# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    # img.save(buf, format="PNG")
    img.save(buf, format="JPG")
    byte_im = buf.getvalue()
    return byte_im



col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    pred = prediction(Image.open(my_upload), image_size = 512)
    col1.image(my_upload)
    col2.image(np.fliplr(np.rot90(pred, 3)))
