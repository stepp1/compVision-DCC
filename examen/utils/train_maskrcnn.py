"""
Trains Mask-RCNN model on 2 GPUs using a pretrained model trained on COCO w/ Copy-Paste augmentations .

Inspired from lazytrain.py

@author: Stepp
"""


from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import random
random.seed(600)
import json
import cv2

import detectron2
from detectron2.structures.boxes import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
setup_logger()

import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_setup,
    default_argument_parser,
    default_writers,
    hooks,
    launch
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

import sys
sys.path.append('../utils')

from train_eval import Evaluation


def create_lazy_cfg(lazy_path, max_iter = 100):
    print(f"Creating {lazy_path}...\n")
    cfg = LazyConfig.load(model_zoo.get_config_file(lazy_path))
    default_setup(cfg, [])
    
    model = cfg.model
    model.backbone.bottom_up.stem.norm = \
    model.backbone.bottom_up.stages.norm = \
    model.backbone.norm = "FrozenBN"
    cfg.model = model
    
    cfg.train.init_checkpoint = 'model_final_bb69de.pkl'
    cfg.model.roi_heads.batch_size_per_image = 64
    cfg.dataloader.train.total_batch_size = 8
    cfg.optimizer.lr = 0.003
    cfg.train.amp.enabled = True
    cfg.train.output_dir = '../logs/mask_rcnn'
    cfg.train.max_iter = max_iter
    cfg.model.roi_heads.num_classes = 13
    cfg.dataloader.train.dataset.names = 'modanet_instance_segmentation_train'
    cfg.dataloader.test.dataset.names = 'modanet_instance_segmentation_test'
    
    return cfg

def do_train(cfg):
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        optimizer=optim,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=True)
    start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret
    
    
def main():
    root_directory = Path("../data")
    images_directory = root_directory / "images"
    annots_directory =  root_directory / "annotations"

    path_to_train = annots_directory / 'modanet_instance_segmentation_train.json'
    path_to_test = annots_directory / 'modanet_instance_segmentation_test.json'


    register_coco_instances('modanet_instance_segmentation_train', {}, path_to_train, images_directory)
    register_coco_instances('modanet_instance_segmentation_test', {}, path_to_test, images_directory)

    MetadataCatalog.get('modanet_instance_segmentation_train').set(
        thing_classes=['bag', 'dress', 'footwear', 'skirt', 'top', 'sunglasses', \
                       'headwear', 'shorts', 'pants', 'belt', 'outer', 'scarf', 'boots']
    )
    new_model_lazy_path = 'new_baselines/mask_rcnn_R_50_FPN_50ep_LSJ.py'
    cfg = create_lazy_cfg(new_model_lazy_path)
    print(do_train(cfg))
    
if __name__ == "__main__":
    launch(main(), 2, 1)