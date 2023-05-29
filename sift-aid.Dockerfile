FROM sift-aid_pre_sift_aid
MAINTAINER Oscar Sjoeberg
LABEL maintainer="oscarbror.sjoberg@gmail.com"

WORKDIR /sift-aid

# COPY ./requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --ignore-installed

## This needs to be done to solve problems with protobuf
RUN pip3 uninstall -y protobuf python3-protobuf
RUN pip3 install --upgrade protobuf

# RUN mkdir -p build && cd build && cmake .. && make


