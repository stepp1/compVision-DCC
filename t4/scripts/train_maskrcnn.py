"""
A main training script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from detectron2 import model_zoo
from utils.trainer import MyTrainer
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA
import detectron2.data.transforms as T
from pathlib import Path

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

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

from detectron2.data.datasets.coco import register_coco_instances, convert_to_coco_dict
from pathlib import Path
import os
import json

def update_coco_json(root_path):

    for json_file in root_path.glob('coco*json'):
        coco_dict = json.load(open(json_file))
        
        new_images = []

        for idx, image_dict in enumerate(coco_dict['images']):
            new_data_path = str(root_path / image_dict['file_name'])
            print(new_data_path)
            image_dict['file_name'] = new_data_path

            new_images.append(image_dict)

        coco_dict['images'] = new_images

        kw = str(json_file).split('_')[1]
        filename = root_path / "{}".format(kw)
        with open(filename,"w") as outfile:
            json.dump(coco_dict, outfile)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = setup_dataset(cfg, args.split)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def setup_dataset(cfg, split):
    """
    Updates config to new dataset.
    """
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    relative_path = Path('../')

    update_coco_json(relative_path)

    if not Path('cocosplit.py').exists() and split:
        os.system('wget https://raw.githubusercontent.com/akarazniewicz/cocosplit/master/cocosplit.py')
    
    if split:
        os.system('python cocosplit.py --having-annotations -s 0.75 coco_train.json train.json val.json')
        register_coco_instances("val", {}, str(relative_path / "val.json"), "")

    register_coco_instances("train", {}, str(relative_path / "train.json"), "")
    register_coco_instances("test", {}, str(relative_path / "test.json"), "")

    cfg.DATASETS.TRAIN = ("train",)

    if split:
        cfg.DATASETS.TEST = ("val",)
    else:
        cfg.DATASETS.TEST = ("test",)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    cfg.TEST.AUG.ENABLED = True
    # cfg.INPUT.FORMAT
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--split", type=str, default='n', help="whether to perform train split")
    args = parser.parse_args()

    if args.split == 'y':
        args.split = True  
    elif args.split == 'n':
        args.split = False
    else:
        raise ValueError('Incorrect arg.split value. Use -y or -n')

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )