# pull official base image
FROM python:3.9

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# copy project
COPY . /usr/src/app/
# COPY ./install_files /usr/src/app/install_files

# install recuired packages 
RUN apt-get update && \
    pip install --upgrade pip && \
    apt-get install -y libgl1 && \
    pip install -r ./install_files/requirements.txt && \
    chmod +x *.sh && \
    chmod +x *.py

# set inference script
ENTRYPOINT ["./model_inference.py"]
CMD ["test_app_video.mp4"]