FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && apt-get update && apt-get install -y python3-dev && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch==1.13.0+cpu torchvision==0.14.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY main.py .
COPY aws_dataset.py .
COPY UNet_model.py .
COPY metric.py .
COPY train.py .

EXPOSE 8000

CMD ["python", "main.py"]