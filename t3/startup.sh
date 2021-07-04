#!/bin/bash
apt update -y && apt-get install -y zip git gcc python3-opencv software-properties-common
git clone https://github.com/stepp1/compVision-DCC.git
pip install gdown; pip install opencv-python

gdown --id 1rw1eoKotPEPhF29vhMYKLOI4ptB-VQ26
gdown --id 1WYaw4LNmRwehoXqaQ7D-a7ZbxBBO5nAh

rm -r compVision-DCC/t3/data
unzip '*.zip' -d compVision-DCC/t3/data

python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
