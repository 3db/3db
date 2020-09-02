FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /blender

ENV PATH="/blender:$PATH"
ENV PYTHONPATH="/code"

RUN mkdir /data; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y wget atool xorg openbox g++ libopenexr-dev git; \
    wget https://mirror.clarkson.edu/blender/release/Blender2.90/blender-2.90.0-linux64.tar.xz; \
    wget https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tgz; \
    aunpack blender-2.90.0-linux64.tar.xz; \
    aunpack Python-3.7.7.tgz; \
    mv blender-2.90.0-linux64/* .; \
    rmdir blender-2.90.0-linux64; \
    rm blender-2.90.0-linux64.tar.xz; \
    rm Python-3.7.7.tgz; \
    cp -r Python-3.7.7/Include/* 2.90/python/include/python3.7m/; \
    rm -rf Python-3.7.7;\
    ./2.90/python/bin/python3.7m -m ensurepip; \
    ./2.90/python/bin/python3.7m -m pip install --upgrade pip; \
    ./2.90/python/bin/python3.7m -m pip install numpy pandas opencv-python git+https://github.com/GuillaumeLeclerc/bpycv
