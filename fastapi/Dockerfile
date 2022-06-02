FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

RUN gpg --keyserver keyserver.ubuntu.com --recv A4B469963BF863CC
RUN gpg --export --armor A4B469963BF863CC | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils


RUN mkdir /fastapi

COPY requirements.txt /fastapi

WORKDIR /fastapi

RUN pip install cython

RUN pip install wheel

RUN pip install numpy

RUN apt-get install gcc

RUN apt-get install libfreetype6-dev -y
RUN apt-get install pkg-config -y

RUN apt install libpython3.7-dev -y

RUN pip install lap

RUN pip install torch==1.7.0 torchvision==0.8.0 

RUN pip install -r requirements.txt 

COPY . /fastapi


RUN apt-get install -y software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake \
    bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion libopenmpi-dev && \
    apt-get clean

RUN add-apt-repository ppa:savoury1/build-tools
RUN add-apt-repository ppa:savoury1/backports
RUN add-apt-repository ppa:savoury1/fonts
RUN add-apt-repository ppa:savoury1/graphics
RUN add-apt-repository ppa:savoury1/multimedia
RUN add-apt-repository ppa:savoury1/ffmpeg4
RUN apt-get update
RUN apt-get install ffmpeg -y

WORKDIR  /fastapi/DCNv2-pytorch_1.7

RUN apt-get update && apt-get install libtinfo5

RUN  bash make.sh

WORKDIR ../


EXPOSE 8501

CMD ["streamlit", "run", "ui.py","mot"]