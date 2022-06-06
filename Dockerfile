FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

RUN mkdir /workspace
COPY * /workspace/cdd4dusia
WORKDIR /workspace/cdd4dusia

#ENV PATH /module:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin