FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

WORKDIR /blender

ENV PATH="/blender:$PATH"
ENV PYTHONPATH="/code"

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y wget atool xorg openbox \
        g++ libopenexr-dev git build-essential subversion cmake nodejs npm \
        libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev \
        libxinerama-dev libglew-dev clang-9 libc++-9-dev libc++abi-9-dev \
        cmake ninja-build libz-dev libpng-dev libjpeg-dev libxrandr-dev zsh \
        libxinerama-dev libxcursor-dev ffmpeg wget imagemagick htop tmux 
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Mitsuba install
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
RUN wget "https://www.dropbox.com/s/3h94dh92lk1mb05/optix7_installer?raw=1" -O optix7_installer
RUN bash optix7_installer --skip-license --prefix=/root --include-subdir
RUN rm -f optix7_installer

COPY docker_requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN rm -f /requirements.txt
RUN wget https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl; \
    pip install bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install; \
    rm -f bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl;

ENV CC=clang-9
ENV CXX=clang++-9
ENV CUDACXX=/usr/local/cuda/bin/nvcc

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
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN echo "export THEME=agnoster" >> ~/.zshrc

RUN wget https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt\?raw=1 -O /cifar_nat.pt 
WORKDIR /

RUN echo "set-option -g default-shell /bin/zsh" >> /etc/tmux.conf

CMD ["zsh"]