#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

# conda install -c omgarcia gcc-6 # install GCC version 6
# conda install libgcc


# In[2]:


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer


import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], 'detr'))

from detr.d2.train_net import add_detr_config, Trainer

# In[3]:


from detectron2.data.datasets.coco import register_coco_instances, convert_to_coco_dict

from pathlib import Path

data_path = Path('data/SpermSegGS/')

if not Path('cocosplit.py').exists():
    get_ipython().system('wget https://raw.githubusercontent.com/akarazniewicz/cocosplit/master/cocosplit.py')
get_ipython().system('python cocosplit.py --having-annotations -s 0.75 coco_train.json train.json val.json')

register_coco_instances("sperm-train", {}, "train.json", "")
register_coco_instances("sperm-val", {}, "val.json", "")
register_coco_instances("sperm-test", {}, "coco_test.json", "")


# In[4]:


# augmentation ...


# In[5]:


# https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl


# In[6]:

def main():
    cfg = get_cfg()
    add_detr_config(cfg)

    cfg.merge_from_file('detrd2/configs/detr_segm_256_6_6_torchvision.yaml')
    cfg.DATASETS.TRAIN = ("sperm-train",)
    cfg.DATASETS.TEST = ("sperm-test",)
    cfg.TEST.EVAL_PERIOD = 250
    cfg.DATALOADER.NUM_WORKERS = 9
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth'
    cfg.SOLVER.IMS_PER_BATCH = 1
    # cfg.SOLVER.BASE_LR = 0.00020  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 2500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset

    # cfg.SOLVER.STEPS = []        # do not decay learning rate

    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32 
    # # see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()