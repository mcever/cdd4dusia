#docker run -it --ipc="host" --gpus all -v /media/ssd1/connor/frames/:/workspace/cdd4dusia/data/frames cdd4dusia

FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt-get update --fix-missing -y
RUN apt-get install ffmpeg libsm6 libxext6  gcc vim -y

COPY . /workspace/cdd4dusia/
WORKDIR /workspace/cdd4dusia
RUN pip install -r requirements.txt