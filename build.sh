#!/bin/bash

# echo 'Building Dockerfile with image name monop'
# docker build \
#     -t monop \
#     -f Dockerfile \
#     .
#     # --build-arg REQS="$(cat requirements.txt | tr '\n' ' ')" \
#     # --build-arg UID=$(id -u ${USER}) \
#     # --build-arg GID=1234 \


MYUSER=myuser
DOCKER_IMAGE_NAME=monop
IMAGE=${DOCKER_IMAGE_NAME}:latest
# export USE_CUDA=$(if $(GPUS),true,false)
USE_CUDA=true
ID=$(id -u)

export DOCKER_BUILDKIT=1
docker build \
    --build-arg USE_CUDA=${USE_CUDA} \
    --build-arg MYUSER=${MYUSER} \
    --build-arg UID=${ID} \
    --tag ${IMAGE} \
    --progress=plain \
    ${PWD}/.