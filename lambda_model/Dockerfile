FROM public.ecr.aws/lambda/python:3.9
COPY requirements.txt .
RUN python -m pip install --upgrade pip && yum -y clean all
# RUN yum install gcc -y
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" && yum -y clean all
RUN pip3 install torch==1.13.0+cpu torchvision==0.14.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && yum -y clean all

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY UNet_model.py ${LAMBDA_TASK_ROOT}
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.lambda_handler" ]