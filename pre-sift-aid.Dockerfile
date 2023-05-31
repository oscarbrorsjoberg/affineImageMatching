FROM tensorflow/tensorflow:1.8.0-gpu-py3
LABEL maintainer="oscarbror.sjoberg@gmail.com"

## replace with git pull?
RUN mkdir /sift-aid 
WORKDIR /sift-aid

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/7fa2af80.pub

## This takes up alot 
## What can I remove?

RUN apt-get update && \
    apt-get upgrade -y --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated \
    cmake \
    pkg-config \
    python3-tk \
    libpng-dev \
    libfreetype6-dev \
    libboost-all-dev \
    libsm6 libxext6 \
    wget \
    build-essential \ 
    unzip \ 
    libopencv-dev \ 
    libjpeg-dev \ 
    libtiff-dev \ 
    libgtk2.0-dev \ 
    libatlas-base-dev \
    gfortran \
    webp \ 
    qt5-default \
    libvtk6-dev \ 
    zlib1g-dev 

COPY ./sift-aid/ /sift-aid/
RUN mkdir -p build && cd build && cmake .. && make

## make this one image
# COPY ./requirements.txt /requirements.txt
# RUN pip3 install --upgrade pip
# RUN pip3 install -r requirements.txt --ignore-installed
## This needs to be done to solve problems with protobuf
# RUN pip3 uninstall protobuf python3-protobuf
# RUN pip3 install --upgrade protobuf



