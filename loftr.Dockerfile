# Use nvidia/cuda image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer="oscarbror.sjoberg@gmail.com"

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# https://askubuntu.com/questions/1444943/nvidia-gpg-error-the-following-signatures-couldnt-be-verified-because-the-publi
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 \ 
    libxrender1 git mercurial subversion \ 
    ffmpeg libsm6 libxext6 && \
        apt-get clean

# install anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

RUN mkdir /app
WORKDIR /app

# setup conda virtual environment
COPY ./LoFTR/environment.yaml /app/environment.yaml
COPY ./LoFTR/requirements.txt /app/requirements.txt

# RUN conda update conda \
#     && conda env create -f ./environment.yaml

# RUN echo "conda activate loftr" >> ~/.bashrc

# ENV PATH /opt/conda/envs/loftr/bin:$PATH
# ENV CONDA_DEFAULT_ENV $loftr



