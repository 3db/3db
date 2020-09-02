# synthetic-sandbox

## Running

We assume that the blender environments are in `$BLENDER_ENVS` and models are in `$BLENDER_MODELS`

### Master node

- `export PYTHONPATH=$(pwd)`
- `python sandbox/main.py $BLENDER_ENVS $BLENDER_MODELS examples/simple_grid_search.yaml 5555`

### Render node
#### With docker

- `docker build -t ./docker`
- `bash ./docker/run_renderer.sh $BLENDER_ENVS $BLENDER_MODELS`
- `/blender/blender --python-use-system-env -b -P /code/sandbox/client.py -- /data/environments/ /data/models/`

#### Wihtout docker

You are a grown up you can read the docker files and figure it out. Or run with docker :P
