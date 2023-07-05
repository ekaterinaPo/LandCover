FROM python:3.8

WORKDIR /app

COPY requirements.txt main.py aws_dataset.py UNet_model.py metric.py train.py ./

RUN python -m pip install --upgrade pip && apt-get update && apt-get install -y python3-dev && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch==1.13.0+cpu torchvision==0.14.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["python", "main.py"]