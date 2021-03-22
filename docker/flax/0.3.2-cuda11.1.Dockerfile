FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.1

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
    && python -m pip install --upgrade jax jaxlib==0.1.64+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    && pip install flax==0.3.2
