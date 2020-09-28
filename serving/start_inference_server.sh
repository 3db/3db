
MODEL_WEIGHTS_PATH="$1"
MODEL_CODE_PATH="$2"

TEMP_FOLDER="/tmp/synthetic-sandbox/serve"
CONTAINER_MODEL_NAME="sandbox-model"
CONTAINER_MODEL_WEIGHTS_PATH="/home/model-server/examples/image_classifier/weights.pth"
CONTAINER_CODE_PATH="/home/model-server/examples/image_classifier/code.py"
CONTAINER_STORE_PATH="/home/model-server/model-store"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "[Cleaning previous runs]"
rm -rf $TEMP_FOLDER

echo "[Building torchserve]"
DOCKER_BUILDKIT=1 docker build --file Dockerfile --network=host --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 -t torchserve:latest . 2> /dev/null

mkdir -p $TEMP_FOLDER

CONTAINER_ID=$(docker run -d --rm -p 8080:8080 -p 8081:8081 --network=host -v $TEMP_FOLDER:/home/model-server/model-store -v $DIR/serve/examples:/home/model-server/examples  torchserve:latest)


echo "[Copying model weights+code]"

docker cp $MODEL_WEIGHTS_PATH $CONTAINER_ID:$CONTAINER_MODEL_WEIGHTS_PATH
docker cp $MODEL_CODE_PATH $CONTAINER_ID:$CONTAINER_CODE_PATH

echo "[Compiling model]"
docker exec $CONTAINER_ID \
    torch-model-archiver \
        --model-name $CONTAINER_MODEL_NAME \
        --version 1.0 \
        --model-file $CONTAINER_CODE_PATH \
        --serialized-file $CONTAINER_MODEL_WEIGHTS_PATH \
        --export-path $CONTAINER_STORE_PATH \
        --extra-files /home/model-server/examples/image_classifier/index_to_name.json \
        --handler image_classifier

echo "[Stopping conversion container]"
docker stop -t 0 $CONTAINER_ID

echo "[Start serving]"
docker run --rm -it \
    -p 8080:8080 \
    -p 8081:8081 \
    --network=host \
    --ipc=host \
    -v $TEMP_FOLDER:$CONTAINER_STORE_PATH \
    -v $DIR/serve/examples:/home/model-server/examples torchserve:latest torchserve --start --ncs --model-store $CONTAINER_STORE_PATH --models $CONTAINER_MODEL_NAME.mar

echo "[Cleaning up]"

rm -rf $TEMP_FOLDER
