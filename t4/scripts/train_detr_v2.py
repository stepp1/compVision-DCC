"""
DETR Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys
import itertools

# fmt: off
import os

# Get the current working directory
cwd = os.chdir  ()
sys.path.insert(1, os.path.join(sys.path[0], '../detr'))
# fmt: on

import time
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from d2.train_net import add_detr_config, DetrDatasetMapper

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from detectron2.solver.build import maybe_add_gradient_clipping


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if "Detr" == cfg.MODEL.META_ARCHITECTURE:
            mapper = DetrDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.merge_from_file('../detr/d2/configs/detr_segm_256_6_6_torchvision.yaml')
    cfg.merge_from_list(args.opts)
    cfg = setup_dataset(cfg, args.split)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


from detectron2.data.datasets.coco import register_coco_instances, convert_to_coco_dict
from pathlib import Path
import os

def update_coco_json(data_path):
    root_path = Path(data_path.parts[0])

    for json_file in root_path.glob('coco*json'):
        coco_dict = json.load(open(json_file))

        for idx, image_dict in coco_dict['images']:
            new_data_path = image_dict['file_name'].replace('data/', str(data_path))
            image_dict['file_name']
            coco_dict['images'][idx] = image_dict

        with open("train.json".format(keyword),"w") as outfile:
            json.dump(coco_dict, outfile)

def setup_dataset(cfg, split):
    """
    Updates config to new dataset.
    """
    relative_path = Path('../data/')

    update_coco_json(data_path)

    if not Path('cocosplit.py').exists() and split:
        os.system('wget https://raw.githubusercontent.com/akarazniewicz/cocosplit/master/cocosplit.py')
    
    if split:
        os.system('python cocosplit.py --having-annotations -s 0.75 coco_train.json train.json val.json')
        register_coco_instances("val", {}, "val.json", "")

    register_coco_instances("train", {}, "train.json", "")
    register_coco_instances("test", {}, "test.json", "")

    cfg.DATASETS.TRAIN = ("train",)

    if split:
        cfg.DATASETS.TEST = ("test",)
    else:
        cfg.DATASETS.TEST = ("val",)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    # cfg.INPUT.FORMAT
    return cfg



def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
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


    # python train_detr_v2.py --num-gpus 2