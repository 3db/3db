#!/bin/bash

code_folder=$(dirname "$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )")
env_folder=$(realpath $1)
models_folder=$(realpath $2)

echo $SCRIPTPATH

docker run -it --network=host --ipc=host --rm  \
    --mount type=bind,source="$code_folder",target=/code \
    --mount type=bind,source="$env_folder",target=/data/environments \
    --mount type=bind,source="$models_folder",target=/data/models \
    sandbox bash
