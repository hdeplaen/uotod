# HAND CRAFTED EXAMPLE FOR ILLUSTRATION PURPOSE

import numpy
from PIL import Image
import torch


import os
print(os.getcwd())

with Image.open('img2.jpg') as pil_img:
    img = numpy.asarray(pil_img)


num_classes = 2

boxes_pred = torch.tensor([[130., 160., 320., 320.],
                           [350., 280., 550., 400.],
                           [105., 130., 265., 260.],
                           [300., 105., 570., 295.],
                           [ 20.,  50., 120., 120.]])

cls_pred = torch.tensor([[0.2, 0.7, 0.1],
                        [0.1, 0.05, 0.85],
                        [0.1, 0.05, 0.85],
                        [0.2, 0.7, 0.1],
                        [0.1, 0.05, 0.85],], dtype=torch.float)

boxes_target = torch.tensor([[292.62, 134.59, 518.84, 285.05],
                             [114.35, 148.97, 341.14, 297.63]])

cls_target = torch.tensor([0, 1], dtype=torch.long)

mask_target = torch.tensor([True, True], dtype=torch.bool)

input = {
    'pred_logits': cls_pred.unsqueeze(0),
    'pred_boxes': boxes_pred.unsqueeze(0),
}
target = {
    'labels': cls_target.unsqueeze(0),
    'boxes': boxes_target.unsqueeze(0),
    'mask': mask_target.unsqueeze(0)
}

# move to cuda
if torch.cuda.is_available():
    input = {k: v.to('cuda') for k, v in input.items()}
    target = {k: v.to('cuda') for k, v in target.items()}
