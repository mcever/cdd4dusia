# Paper title

This is the code for XX found at XX


# Setup
This project was written and tested using Python 3.7 on Ubuntu 16.04.  You may need as much as 450 GB for all dependencies, code, raw videos, and uncompressed video data. You may need more if you want to work with alternative splits since this requires decompressing some of the video frames.

##  The Code
Clone the repo.

##  Links
Depending on your system, you may need to create soft links. git may do it for you, but it's good to check.

    cd detection
    ln -s ../models mymodels
    ln -s ../mare_utils mare_utils

##  Dependencies
You probably want to set up a virtual environment then run

    pip install -r requirements.txt

##  Data Download
Download the videos from here https://drive.google.com/drive/folders/1vPlA_oswqAEQ2tcQpSrGRoWfKRPeXSpn?usp=sharing and store them in data/idd. Download the splits you want to use form here [https://drive.google.com/drive/u/1/folders/1Hq9USShYSBzxmyiZcbAt8-CbUpQr3IZ6](https://drive.google.com/drive/u/1/folders/1Hq9USShYSBzxmyiZcbAt8-CbUpQr3IZ6) and store them in data/idd_lsts. 

## Data Preparation
Training the model requires decompressing and saving frames from the original video to allow for fast training. This will require a significant amount of storage as it pulls frames from compressed video and saves them uncompressed in .npy files. Due to descrepencies in different video codecs and video seeking, it is highly recommended to use this repo's scripts to pull frames to ensure the pulled frames' numbers match those in the annotations.

    python pull_frames_from_lst path/to/lst path/to/frames

/data/MARE/split_setname is the recommended path/to/frames

## Train the Model

    cd detection
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py    --dataset mare --model fasterrcnn_resnet50_fpn --epochs 25    --lr-steps 16 22 --aspect-ratio-group-factor 3 --output-dir path/to/output --batch-size 12 --workers 4 --lr 0.002 --distributed 0
Run with CUDA_VISIBLE_DEVICES set to whatever you want
exps/outXXX is the recommended path/to/output where XXX is unique to each experiment since detection/exps is included in .gitignore.
