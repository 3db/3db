# synthetic-sandbox

## Running

We assume that the blender environments and models are in `$BLENDER_DATA/blender_models` and `$BLENDER_DATA/blender_environments` 

### Master node

- `export PYTHONPATH=$(pwd)`
- python sandbox/main.py $BLENDER_DATA examples/very_simple_grid_search.yaml 5555 --logdir logs/ --loggers JSONLogger,TbLogger

### Render node
#### With docker

- `docker build -t sandbox ./docker`
- `bash ./docker/run_renderer.sh $BLENDER_DATA`
- `cd /code`
- `bash start_sandbox.sh`

3 Tmux sessions running the client, server, and tensorboard will be started for you!

## Basic Data Collection
This shows you how to collect a dataset using blender to analyze the quality of the models generated. We assume that the blender environments are in `$BLENDER_ENVS`, models are in `$BLENDER_MODELS`, and the desired output directory is in `$OUTDIR`.

- Example enviroment is [here](environments/).
- 3DModels can be downloaded from azure storage:
    - Get `azcopy` which will be usefull to download the auto-generated dataset later on too.
        ```
        wget -O azcopy.tar https://aka.ms/downloadazcopy-v10-linux \
        tar -xzvf azcopy.tar && cp azcopy_linux_amd64_10.6.0/azcopy ~/ && rm -rf azcopy_linux_amd64_10.6.0 azcopy.tar
        ```
    - Run the following command to download the models:
        ```
        azcopy cp 'https://objectsim.blob.core.windows.net/sandbox/blender_models?sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2021-09-04T17:58:15Z&st=2020-09-04T09:58:15Z&spr=https,http&sig=CPuCBI%2FtMrSlzytVt7UioVkyKC9%2Fetp5XqTC2rtjino%3D' '/models' --recursive
        ```
### With docker

- `docker build -t ./docker`
- `bash ./docker/run_renderer.sh $BLENDER_ENVS $BLENDER_MODELS $OUTDIR`
- `/blender/blender -b -P /code/rendering/render.py -- --output=/output/ --models=/data/models/ --log=/output/log.log --metadata=/code/data_collection/output.csv --env=/data/environments/white_background.blend --samples=128 --gpu-id=0 --resolution=256 --resume`

### Without docker
Follow the instruction inside the [Dockerfile](./docker/Dockerfile) to setup things locally, and run the above commands without running docker.
