from detectron2.utils.visualizer import Visualizer, ColorMode
import detectron2.data.transforms as T
import numpy as np

import torch
from utils.trainer import InvertColors

augs = T.AugmentationList(
    [
        InvertColors(),
        T.Resize((300,800)), 
#         T.RandomContrast(1.5, 2.5),
        T.PadTransform(100, 100, 100, 100),
    ]
)


def augment(im):
    input = T.AugInput(im)
    transform = augs(input)  # type: T.Transform
    x = input.image  # new image
    
    return x

def sort_predictions(outputs):
    pred_classes = []
    scores = []
    for out in outputs:
        idxs = np.argsort(out["instances"].pred_boxes.tensor.to('cpu')[:,0])
        pred_classes.append(out["instances"].pred_classes[idxs])
        scores.append(out["instances"].scores[idxs])
    
    return pred_classes, scores

def batched_viz(batch, outputs):
    for im, out in zip(batch, outputs):
        v = Visualizer(im[:, :, ::-1],
                   metadata=checks_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        img_ready = v.draw_instance_predictions(out["instances"].to("cpu"))
        showarray(img_ready.get_image()[:, :, ::-1])

def compute_amounts(preds):
    amounts = []
    
    for pred in preds:
        if isinstance(pred, torch.Tensor):
            pred = pred.to('cpu').numpy()
        pred = pred[::-1]
        amount = np.sum([x * 10 ** i for i, x in enumerate(pred)])
        amounts.append(amount)
        
    return amounts