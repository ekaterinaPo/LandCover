# Base image
FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch==1.13.0+cpu torchvision==0.14.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && yum -y clean all

COPY main.py .
COPY dataset.py .
COPY UNet_model.py .
COPY metrics.py .
COPY train.py .

EXPOSE 8000

CMD ["python", "main.py"]