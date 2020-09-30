#!/bin/sh

# Reading arguments
MODEL_WEIGHTS_PATH=$(realpath "$1")
MODEL_CODE_PATH=$(realpath "$2")
DEVICES=$3
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TORCHSERVER_FOLDER="$DIR/serve/docker"
EXTRA_RUN_ARGS=""

TEMP_FOLDER="/tmp/synthetic-sandbox/serve"
CONTAINER_MODEL_NAME="sandbox-model"
CONTAINER_MODEL_WEIGHTS_PATH="/home/model-server/examples/image_classifier/weights.pth"
CONTAINER_CODE_PATH="/home/model-server/examples/image_classifier/code.py"
CONTAINER_STORE_PATH="/home/model-server/model-store"
MAX_CLASSES=10000


# Checking arguments and environment
if [[ ! -f "$MODEL_WEIGHTS_PATH" ]]
then
    echo "[ERROR] $MODEL_WEIGHTS_PATH not found";
    exit 1;
else
    echo "[INFO] Using weights: $MODEL_WEIGHTS_PATH";
fi

if [[ ! -f "$MODEL_CODE_PATH" ]]; then
    echo "[ERROR] $MODEL_CODE_PATH not found";
    exit 1;
else
    echo "[INFO] Using architecture: $MODEL_CODE_PATH";
fi

if [[ ! -d "$TORCHSERVER_FOLDER" ]]; then
    echo "[ERROR] torchserve not found";
    echo "[INFO] expected folder $TORCHSERVER_FOLDER not found";
    echo "[INFO] You may have forgotten to checkout git submodules";
    exit 1;
else
    echo "[INFO] Using torchserve docker folder: $TORCHSERVER_FOLDER";
fi

if [[ ! -z $DEVICES ]]; then
    echo "[INFO] using GPU devices $DEVICES"
    EXTRA_RUN_ARGS="--gpus \"device=$DEVICES\""
fi

echo "[INFO] Cleaning previous runs"
rm -rf $TEMP_FOLDER

echo "[INFO] Building torchserve"
cd "$DIR/serve/docker"
DOCKER_BUILDKIT=1 docker build --file Dockerfile --network=host --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 -t torchserve:latest .

mkdir -p $TEMP_FOLDER


CONTAINER_ID=$(docker run -d --rm -p 8080:8080 -p 8081:8081 --user $(id -u):$(id -g) --network=host -v $TEMP_FOLDER:/home/model-server/model-store -v $DIR/serve/examples:/home/model-server/examples  torchserve:latest)


echo "[INFO] Copying model weights+code"

docker cp $MODEL_WEIGHTS_PATH $CONTAINER_ID:$CONTAINER_MODEL_WEIGHTS_PATH
docker cp $MODEL_CODE_PATH $CONTAINER_ID:$CONTAINER_CODE_PATH

echo "[INFO] Generating class file"

CODE="import json;print(json.dumps({str(v):str(v) for v in range($MAX_CLASSES)}))"

docker exec $CONTAINER_ID \
    bash -c "python -c \"$CODE\" > /tmp/index_to_name.json"

echo "[INFO] Compiling model"
docker exec $CONTAINER_ID \
    torch-model-archiver \
        --model-name $CONTAINER_MODEL_NAME \
        --version 1.0 \
        --model-file $CONTAINER_CODE_PATH \
        --serialized-file $CONTAINER_MODEL_WEIGHTS_PATH \
        --export-path $CONTAINER_STORE_PATH \
        --extra-files /tmp/index_to_name.json \
        --handler image_classifier

echo "[INFO] Stopping conversion container"
docker stop -t 0 $CONTAINER_ID

echo "[INFO] Start serving"
docker run --rm -it \
    -p 8080:8080 \
    -p 8081:8081 \
    --network=host \
    $EXTRA_RUN_ARGS \
    --ipc=host \
    -v $TEMP_FOLDER:$CONTAINER_STORE_PATH \
    -v $DIR/serve/examples:/home/model-server/examples \
    torchserve:latest torchserve --start --ncs --model-store $CONTAINER_STORE_PATH --models $CONTAINER_MODEL_NAME.mar

echo "[INFO] Cleaning up"

rm -rf $TEMP_FOLDER
