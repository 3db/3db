FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

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
    ./2.90/python/bin/python3.7m -m pip install numpy pandas opencv-python git+https://github.com/GuillaumeLeclerc/bpycv pyzmq torch torchvision

# Mitsuba install
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
RUN wget "https://www.dropbox.com/s/3h94dh92lk1mb05/optix7_installer?raw=1" -O optix7_installer
RUN bash optix7_installer --skip-license --prefix=/root --include-subdir

RUN apt update && apt install -y git \
  clang-9 libc++-9-dev libc++abi-9-dev cmake ninja-build \
  libz-dev libpng-dev libjpeg-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  python3-dev python3-distutils python3-setuptools ffmpeg \
  python3-pytest python3-pytest-xdist python3-numpy wget imagemagick \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV CC=clang-9
ENV CXX=clang++-9
ENV CUDACXX=/usr/local/cuda/bin/nvcc

RUN /opt/conda/bin/python3.7 -m pip install pytest pytest-xdist
RUN git clone --recursive https://github.com/mitsuba-renderer/mitsuba2.git /mitsuba2
ENV PATH=/usr/local/cuda/:/usr/local/cuda/include:${PATH}

WORKDIR /mitsuba2
RUN git pull origin master && git submodule update
COPY mitsuba.conf /mitsuba2/mitsuba.conf
RUN mkdir build \
    && cd build \
    && cmake -GNinja .. \ 
    && ninja

ENV PYTHONPATH=/mitsuba2/dist/python:/mitsuba2/build/dist/python:$PYTHONPATH
ENV PATH=/mitsuba2/dist:/mitsuba2/build/dist:$PATH

RUN echo 'source /mitsuba2/setpath.sh' >> ~/.zshrc
RUN echo 'cd /' >> ~/.zshrc

# Install Zsh, tmux, ipdb, etc. to make life easier
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "zsh"]
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN echo "export THEME=agnoster" >> ~/.zshrc

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y wget libopenexr-dev git
RUN apt-get install -y htop
RUN pip install cox robustness kornia scikit-image orjson \
    opencv-python imgaug zmq OpenEXR tensorboard ipdb flatten-dict
RUN /blender/2.90/python/bin/pip install tqdm ipdb
RUN wget https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt\?raw=1 -O /cifar_nat.pt 
WORKDIR /

RUN apt-get install -y tmux
RUN echo "set-option -g default-shell /bin/zsh" >> /etc/tmux.conf

# Setup for dashboard
RUN pip install flask flask_cors
RUN apt update && apt install -y nodejs npm
CMD ["zsh"]