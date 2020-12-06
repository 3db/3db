#!/bin/bash
code_folder=$(dirname "$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )")
data_folder=$(realpath $1)

docker run -it --network=host --ipc=host --rm  \
    --mount type=bind,source="$code_folder",target=/code \
    --mount type=bind,source="$data_folder",target=/data \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" sandbox