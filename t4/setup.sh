#!/bin/bash
apt-get install -y zip git gcc python3-opencv software-properties-common wget nano

wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -p ./miniconda
./miniconda/bin/conda init

source .bashrc

conda activate base
conda install mamba -n base -c conda-forge -y

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install gdown opencv-python Pillow numpy matplotlib jupyterlab scipy shapely scikit-learn funcy
 
gdown --id 1vkP2gj_lpZDMUU3a8TNRs4rRoJHsbiMd

git clone https://github.com/stepp1/compVision-DCC.git
unzip '*.zip' -d compVision-DCC/t4/data

jupyter-notebook --no-browser -p=8080