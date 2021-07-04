from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T

import torch
import copy
import os

from .hooks import ValidationHook

class InvertColors(T.Augmentation):
    def get_transform(self, image):
        return T.ColorTransform(lambda x: 255-x)

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.Resize((300,800)),
        InvertColors(),
        T.PadTransform(10, 10, 10, 10),
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


class MyTrainer(DefaultTrainer):
    def __init__(self, cfg, val_loss=True, bs=64):
        self._val_loss = val_loss
        self._bs = bs
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, ValidationHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            ),
            patience = 3
        ))
        return hooks
#     def build_hooks(self):
#         hooks = super().build_hooks()
#         if True:
#             hooks.insert(
#                 -1,
#                 ValidationHook(
#                     self.cfg.TEST.EVAL_PERIOD,
#                     self.model,
#                     build_detection_train_loader(
#                         DatasetCatalog.get(self.cfg.DATASETS.TEST[0]),
#                         mapper=DatasetMapper(self.cfg, is_train=True),
#                         total_batch_size=self._bs,
#                     ),
#                 ),
#             )
#         return hooks


class BatchPredictor(DefaultPredictor):
    """Run d2 on a list of images."""

    def __call__(self, images):
        """Run d2 on a list of images.

        Args:
            images (list): BGR images of the expected shape: 720x1280
        """
        images = [
            {"image": torch.from_numpy(image.astype("float32").transpose(2, 0, 1))}
            for image in images
        ]
        with torch.no_grad():
            preds = self.model(images)
        return preds


