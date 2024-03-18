#!/bin/sh

# dirname=$(pwd | xargs dirname)
# dataset="/share/private/27th/horiike/dnn_models/"
dnn_dir="/share/private/27th/horiike/dnn_models/"

docker run -it \
  --privileged \
  --gpus all \
  -p 15900:5900 \
  --rm \
  # --mount type=bind,source=$dirname,target=/root/relpose \
  # --mount type=bind,source=$dataset,target=/root/dataset \
  --mount type=bind,source=$dnn_dir,target=/root/ \
   --net host \
   --shm-size=40gb \
  # relpose
  bash
