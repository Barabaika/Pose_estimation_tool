# pull official base image
FROM tensorflow/tensorflow:latest
# tensorflow:latest
# FROM python:3.9

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# copy project
COPY . /usr/src/app/

# install recuired packages 
RUN apt-get update && \
    pip install --upgrade pip && \
    apt-get install -y libgl1 && \
    apt-get install -y libglib2.0-0 && \
    pip install -r ./help_files/requirements.txt && \
    chmod +x *.sh && \
    chmod +x ./help_files/*.py

# set inference script
ENTRYPOINT ["./main/model_inference.py"]
CMD ["test_app_video.mp4"]