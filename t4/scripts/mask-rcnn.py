#!/usr/bin/env python
# coding: utf-8


import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

# conda install -c omgarcia gcc-6 # install GCC version 6
# conda install libgcc



import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os 

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T

from utils.trainer import MyTrainer

from detectron2.data.datasets.coco import register_coco_instances, convert_to_coco_dict

from pathlib import Path

data_path = Path('data/SpermSegGS/')


# In[1]:

class InvertColors(T.Augmentation):
    def get_transform(self, image):
        return T.ColorTransform(lambda x: 255-x)

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        InvertColors(),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomLighting(0.7),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class Trainer(MyTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, ('segm',), False, output_folder)


# In[10]:

def main():
    if not Path('cocosplit.py').exists():
    get_ipython().system('wget https://raw.githubusercontent.com/akarazniewicz/cocosplit/master/cocosplit.py')
    get_ipython().system('python cocosplit.py --having-annotations -s 0.75 coco_train.json train.json val.json')

    register_coco_instances("sperm-train", {}, "train.json", "")
    register_coco_instances("sperm-val", {}, "val.json", "")
    register_coco_instances("sperm-test", {}, "coco_test.json", "")


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("sperm-train",)
    cfg.DATASETS.TEST = ("sperm-test",)
    cfg.TEST.EVAL_PERIOD = 250
    cfg.DATALOADER.NUM_WORKERS = 9
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_50_FPN_200ep_LSJ/42047638/model_final_89a8d3.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00010  # pick a good LR
    cfg.SOLVER.MAX_ITER = 2500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset

    # cfg.SOLVER.STEPS = []        # do not decay learning rate

    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32 
    # # see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


# In[6]:


    torch.save(trainer.model.state_dict(), './output/model_final.pth')


# In[7]:
import io
import cv2
import numpy as np
from IPython.display import clear_output, Image, display
from PIL import Image

import torchvision.transforms.functional as TF

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()

    display(Image.fromarray(a))

import json
import random

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode


def predict():
    from detectron2.engine import DefaultPredictor

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = json.load(open('coco_test.json'))['images']
    dataset_metadata = Metadat aCatalog.get("train")

    for d in random.sample(dataset_dicts, 4):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=dataset_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        showarray(out.get_image())  


if __name__ == '__main__':
    main()



