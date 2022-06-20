# Context-Driven Detection of Invertebrate Species in Deep-Sea Video

This is the code for the CVPR22 CV4Animal's poster.

# Setup
This project was written and tested using Python 3.7 on Ubuntu 16.04.  You may need as much as 450 GB for all dependencies, code, raw videos, and uncompressed video data. You may need more if you want to work with alternative training splits since training requires decompressing some of the video frames.

##  Data Download
The data will be available on BisQue. To download the data see instructions here https://www.austinmcever.com/CDD4DUSIA#h.796kvbv3ui3r


# Docker
In order to get all the correct versions of dependencies, we recommend running this code using our provided Docker container. To do so follow these steps:

        
* Make sure you have Docker installed

        docker version
        
* You can pull a pre-built container from TBD 

        docker TBD

* Build the CDD4Dusia Image

        docker build -t cdd4dusia .

* If you only have the videos downloaded, run this command to start the container and continue to "Data Preparation"

        docker run -it --ipc="host" --gpus all -v /ABSOLUTE-PATH-TO-VIDEOS/:/workspace/cdd4dusia/data/vids cdd4dusia

  * Note: If you have already extracted the frames with annotations for training, mount the frames to /workspace/cdd4dusia/data/frames in a similar fashion
  * If you do not have nvidia-docker installed (highly recommended), do  not include the GPU argument

        

# Running without Docker
Not all systems and Python versions are compatible with this code. This project was written and tested using Python 3.7 on Ubuntu 16.04. Begin by cloning the code and ensuring all soft links are properly handled.

##  Config File
Inside of *config.yaml* make sure that the paths to data directory and the directory of downloaded videos is correct

##  Dependencies
You probably want to set up a virtual environment before you run

    pip install -r requirements.txt

# Data Preparation
Training the model requires decompressing and saving frames from the original video to allow for fast training. This will require a significant amount of storage as it pulls frames from compressed video and saves them uncompressed in .npy files. Due to discrepancies in different video codecs and video seeking, it is highly recommended to use this repo's scripts to pull frames to ensure the pulled frames' numbers match those in the annotations.

    cd cdd_utils
    python pull_frames_from_lst.py

Note that this will save uncompressed NumPy arrays inside of cdd4dusia/data/frames. If you cloned the repo on a smaller drive (say, an SSD) and would like to store the data on a separate drive, it is recommended to link to cdd4dusia/data/frames to that separate drive before running this command.

# Train the Model

    python train.py
Run with CUDA_VISIBLE_DEVICES set to whatever you want. You can change the hyper-parameters of the model and training script inside of *config.yaml*.
