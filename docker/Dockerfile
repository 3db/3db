FROM continuumio/anaconda3:latest

ENV env_name threedb
RUN apt-get update
ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux
RUN apt-get install -y curl
RUN apt-get install -y build-essential git subversion cmake libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libglew-dev

RUN conda create -y -n $env_name python=3.7.7
RUN echo $env_name
RUN ln -s $(conda run -n $env_name bash -c 'echo \$CONDA_PREFIX') /conda
ENV CONDA_PREFIX /conda
RUN curl "https://mirror.clarkson.edu/blender/release/Blender2.92/blender-2.92.0-linux64.tar.xz" --output /tmp/blender.tar.xz
WORKDIR ${CONDA_PREFIX}
RUN tar -xf /tmp/blender.tar.xz -C ./
RUN rm /tmp/blender.tar.xz
RUN mv blender-2.92.0-linux64/ blender
RUN rm -rf ./blender/2.92/python
RUN ln -s $CONDA_PREFIX ./blender/2.92/python
RUN ln -s $CONDA_PREFIX/blender/blender $CONDA_PREFIX/bin/blender

RUN $CONDA_PREFIX/bin/pip install torch==1.7.0 torchvision cox robustness kornia scikit-image orjson \
                opencv-python zmq tensorboard ipdb flatten-dict tqdm pytest pytest-xdist \
                flask flask_cors tensorboard mathutils typeguard sphinx_rtd_theme pyyaml flask-compress

RUN ln -s /3db/threedb_master $CONDA_PREFIX/bin
RUN ln -s /3db/threedb_workers $CONDA_PREFIX/bin
RUN ln -s /3db/threedb $CONDA_PREFIX/lib/python3.7/site-packages

WORKDIR /
RUN git clone https://github.com/3db/blog_demo.git
RUN apt-get install unzip
RUN cd /blog_demo; bash setup.sh; rm -rf output