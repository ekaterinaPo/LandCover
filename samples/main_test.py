from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.responses import FileResponse, Response
import os
from random import randint
import uuid
import boto3
from PIL import Image
from io import BytesIO


app = FastAPI()

 
IMAGEDIR = "images/"

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.get("/")
def read_root():
    return {"model": "Semantic Segmentation"}
 
@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
 
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
 
    #save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
 
    return {"filename": file.filename}
 
 
@app.get("/show/")
async def read_random_file():
 
    # get random file from the image directory
    files = os.listdir(IMAGEDIR)
    random_index = randint(0, len(files) - 1)
 
    path = f"{IMAGEDIR}{files[random_index]}"
     
    return FileResponse(path)



@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    
    if not extension:
        return "Image must be jpg or png format!"
    
    contents = await file.read()
    image = read_imagefile(contents)

    #buf_image = BytesIO()
    #image.save(buf_image, format=extension)
    #byte_im = buf_image.getvalue()
    base_filename = os.path.splitext(file.filename)[0]
    s3_client = boto3.client('s3', config=boto3.session.Config(
        region_name='us-east-1'
    ))
    s3_image_key = f'{base_filename}.jpg'
    s3_client.put_object(
        Bucket='landcover-prediction',
        Key=s3_image_key,
        Body=contents,
        ContentType=f"image/{extension}"
    )
    
    modified_maskname = base_filename.replace("_sat", "_mask")
    result_filename = f'{modified_maskname}.png'
    result_path = os.path.join(IMAGEDIR, result_filename)
    image.save(result_path, format="PNG")

    #return FileResponse(result_path)
    return {"result_path": result_path}

@app.get("/show_image/")
async def show_image(result_path: str):
    # Example using PIL
    from PIL import Image
    image = Image.open(result_path)

    return image
