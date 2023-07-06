FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

RUN /app/venv/bin/pip install --upgrade pip
RUN apt-get update && apt-get install -y python3-dev && apt-get clean
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt
RUN /app/venv/bin/pip install --no-cache-dir torch==1.13.0+cpu torchvision==0.14.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY main.py aws_dataset.py UNet_model.py metric.py train.py ./

EXPOSE 8000

CMD ["python", "main.py"]