FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update --fix-missing
RUN apt-get install -y git

ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-10.2

RUN apt-get update
RUN apt-get install -y wget

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN python -m pip install --upgrade pip \
    && python -m pip install --upgrade jax jaxlib==0.1.59+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    && pip install flax==0.3.0
