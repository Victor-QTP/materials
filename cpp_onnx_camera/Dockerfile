FROM nvcr.io/nvidia/tensorrt:23.04-py3

RUN sed -i 's@archive.ubuntu.com/ubuntu@free.nchc.org.tw/ubuntu@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com/ubuntu@free.nchc.org.tw/ubuntu@g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopencv-dev=4.2.0+dfsg-5 \
    libglib2.0-0=2.64.6-1~ubuntu20.04.9 \
    gdb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
