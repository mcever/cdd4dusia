#docker run -it --ipc="host" --gpus all -v
#/media/ssd1/connor/frames/:/workspace/cdd4dusia/data/frames cdd4dusia

FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt-get update --fix-missing -y
RUN apt-get install ffmpeg libsm6 libxext6  gcc -y
#do not use pip install -r req.txt for caching reasons
RUN pip install av==8.0.3 \
                Cython==0.29.22 \
                imageio==2.8.0 \
                imageio-ffmpeg==0.4.2 \
                imgaug==0.4.0 \
                ipython==7.14.0 \
                matplotlib==3.2.1 \
                opencv-python-headless==4.5.3.56 \
                pandas==1.0.5 \
                parso==0.7.0 \
                pdbpp==0.10.2 \
                Pillow==7.1.2 \
                pycocotools==2.0.2 \
                scikit-image==0.17.2 \
                scikit-learn==0.24.2 \
                scikit-video==1.1.11 \
                scipy==1.4.1 \
                tornado==6.1 \
                tqdm==4.58.0 \
                urllib3==1.26.3

COPY . /workspace/cdd4dusia/
WORKDIR /workspace/cdd4dusia