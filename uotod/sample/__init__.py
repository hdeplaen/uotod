# HAND CRAFTED EXAMPLE FOR ILLUSTRATION PURPOSE

import numpy
from PIL import Image
import torch
import os

_filedir = os.path.dirname(os.path.realpath(__file__))
_filename = 'img.jpg'
_filepath = os.path.join(_filedir, _filename)

with Image.open(_filepath) as pil_img:
    img = numpy.asarray(pil_img)


num_classes = 2

boxes_pred = torch.tensor([[25, 12, 242, 215],                     # 1
                           [362, 126, 469, 425],                        # 2
                           [87, 241, 221, 343],                         # 3
                           [41, 52, 353, 283],                          # 4
                           [287, 76, 439, 388]], dtype=torch.float)     # 5

cls_pred = torch.tensor([[0.2, 0.7, 0.1],
                        [0.1, 0.05, 0.85],
                        [0.1, 0.05, 0.85],
                        [0.2, 0.7, 0.1],
                        [0.1, 0.05, 0.85],], dtype=torch.float)

boxes_target = torch.tensor([[52, 22, 308, 328],                          # A
                            [26, 106, 456, 394]], dtype=torch.float)      # B

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
anchors = torch.tensor([[0, 0, 200, 200],                      # 1
                        [300, 100, 400, 400],                       # 2
                        [100, 250, 200, 320],                       # 3
                        [50, 50, 350, 300],                         # 4
                        [300, 50, 450, 400]], dtype=torch.float)    # 5

# move to cuda
if torch.cuda.is_available():
    input = {k: v.to('cuda') for k, v in input.items()}
    target = {k: v.to('cuda') for k, v in target.items()}
    anchors = anchors.to('cuda')
