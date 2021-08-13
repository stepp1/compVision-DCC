from imantics import Polygons, Mask
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import random
import cv2
import os

import logging

import remo

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config.config import CfgNode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import load_coco_json
from detectron2.data.catalog import Metadata
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

class Evaluation:
    def __init__(self, cfg, test_set : str, default_pred_cfg : bool = True):
        self.cfg = cfg
        self.config.DATASETS.TEST = (test_set,)
        self.config.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
        self.predictor = None
        if default_pred_cfg:
            self.config.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4 # original 0.5
            self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set the testing threshold for this model

    @property
    def cfg(self):
        if isinstance(self.config, CfgNode):
            return self.config
        return None

    @cfg.setter
    def cfg(self, cfg: CfgNode):
        self.config = cfg
        self.predictor = DefaultPredictor(cfg)

    def run_coco_eval(self):
        trainer = DefaultTrainer(self.config)
        evaluator = COCOEvaluator(self.config.DATASETS.TEST[0], self.config, False, output_dir=self.config.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.config, self.config.DATASETS.TEST[0])
        print(inference_on_dataset(trainer.model, val_loader, evaluator))

    def createAnnots(self, dataset_dicts, mapping, sample_ratio = 1):
        test_annots = []

        for idx, d in tqdm(enumerate(dataset_dicts[::sample_ratio]), total = len(dataset_dicts[::sample_ratio])):
            im = cv2.imread(d['file_name'])
            outputs = self.predictor(im)
            pred_classes = outputs['instances'].get('pred_classes').cpu().numpy()
            masks = outputs['instances'].get('pred_masks').cpu().permute(1, 2, 0).numpy()
            image_name = d['file_name']
            annotations = []
            try:
                if masks.shape[2] != 0:
                    for i in range(masks.shape[2]):
                        polygons = Mask(masks[:, :, i]).polygons()
                        annotation = remo.Annotation()
                        annotation.img_filename = image_name
                        annotation.classes = mapping[pred_classes[i]]
                        annotation.segment = polygons.segmentation[0]
                        annotations.append(annotation)
                elif masks.sum() == 0:
                    continue
                else:
                    polygons = Mask(masks[:, :, 0]).polygons()
                    annotation = remo.Annotation()
                    annotation.img_filename = image_name
                    annotation.classes = mapping[pred_classes[0]]
                    annotation.segment = polygons.segmentation[0]
                    annotations.append(annotation)
            except IndexError:
                raise IndexError(f"No preds at idx: {idx} \n   - instance: \n{d} \n   - outputs: \n{outputs}")
                
            test_annots += annotations

        return test_annots

    def show_n(self, dataset_dicts, metadata, n = 3, predictions = False, add_info = False):
        self.cfg = self.cfg
        f = Evaluation._show_n(dataset_dicts, metadata, n, predictions, predictor=self.predictor, add_info = add_info)
        return f

    @staticmethod
    def _show_n(dataset_dicts, metadata, n = 3, predictions = False, predictor = None, add_info = False):
        f, axs = plt.subplots(
            nrows=2 if n//2 > 2 else 1, 
            ncols=n//2 if n//2 > 2 else n, 
            figsize=(10,8)
            )

        for idx, d in enumerate(random.sample(dataset_dicts, 3)): 
            title = ""
            if add_info: 
                img_fname, n_annots = d["file_name"].split('/')[-1], len(d["annotations"])
                title = f"image: {img_fname} \nGT has {n_annots} annotations"
                
            im = cv2.imread(d["file_name"])
            if predictions:
                out = Evaluation.draw(
                    im, 
                    metadata, 
                    predictions=True, 
                    predictor = predictor, 
                    add_info = add_info
                )
                if add_info:
                    n_preds, out = out["n_annots"], out["out"]
                    title += f"\n Model predicted {n_preds}"
            else:
                out = Evaluation.draw(im, metadata, dataset_dict=d,)
            axs[idx].imshow(out.get_image()[:, :, ::-1])
            axs[idx].axis("off")
            axs[idx].set_title(title)
            
        plt.tight_layout()
        plt.show()
        return f
        
    @staticmethod
    def draw(im : np.ndarray, metadata : Metadata, predictions = False, dataset_dict = None, predictor = None, add_info = False):
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW if predictions else ColorMode.IMAGE
        )

        if predictions:
            outputs = predictor(im) 
            if len(outputs["instances"]) == 0: 
                logger = logging.getLogger("detectron2")
                logger.warning("Model made 0 predictions!")
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out = {"out": out, "n_annots": len(outputs["instances"])} if add_info else out
        elif dataset_dict is not None:
            out = v.draw_dataset_dict(dataset_dict)
        else:
            raise ValueError("Only drawing of predictions or dataset_dict is supported.")
        return out