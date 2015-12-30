#!/bin/bash

DOCKER=${DOCKER:-"docker"}


$DOCKER build -t ann-fab .

GPU=0 nvidia-docker run --rm -v $(pwd):$(pwd) -w $(pwd) -ti ann-fab $*
