ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.1.1
ARG CUDNN_VERSION=8
ARG CUDA_OPSET=devel

FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-$CUDA_OPSET-ubuntu$UBUNTU_VERSION

ENV DEBIAN_FRONTEND="noninteractive"
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.1

# Install apt-get dependencies and Google Cloud SDK, then remove
# files that are not needed to run the container.
RUN apt update --fix-missing && apt upgrade -y \
    && apt install -y build-essential curl ffmpeg libsm6 libxext6 software-properties-common unzip \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

ARG PY_VERSION=3.9
ARG FLAX_VERSION=0.5

RUN add-apt-repository ppa:deadsnakes/ppa && apt update \
    && apt install -y python$PY_VERSION-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python$PY_VERSION 10 \
    && apt install -y python3-setuptools \
    && python -m easy_install pip \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y numpy
RUN pip install --no-cache-dir \
    jax[cuda] \
    jaxlib[cuda11_cudnn82] \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install --no-cache-dir flax~=$FLAX_VERSION.0

COPY ./flaxseed ./flaxseed

# FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04


# RUN apt-get update
# RUN apt-get install -y wget

# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 

# RUN python -m pip install --upgrade pip \
#     && python -m pip install --upgrade jax[cuda] jaxlib[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html \
#     && pip install flax~=0.5.0
