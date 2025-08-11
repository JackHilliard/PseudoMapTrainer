#FROM docker.io/pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
FROM docker.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# Install the required packages
ARG TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update \
    && apt-get install -y whiptail ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 python3.8 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.8 as the default python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Install and activate conda
#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    /bin/bash ~/miniconda.sh -b -p /opt/conda
#
#ENV PATH=$CONDA_DIR/bin:$PATH
#RUN conda env create -n maptr
#SHELL ["conda", "run", "-n", "maptr", "/bin/bash", "-c"]
#ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.0

RUN pip install timm

RUN pip install networkx==2.2 \
 && pip install scikit-image==0.19.0 \
 && pip install numpy==1.23.5 \
 && pip install scikit-learn==1.3.0 \
 && pip install yapf==0.40.1 \
 && pip install pandas \
 && pip install ipython

# Install MMDetection3D
#RUN conda clean --all
COPY . /MapTR
WORKDIR /MapTR/mmdetection3d
RUN python setup.py develop

WORKDIR /MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
RUN python setup.py build install

WORKDIR /MapTR
RUN pip install -r requirement.txt