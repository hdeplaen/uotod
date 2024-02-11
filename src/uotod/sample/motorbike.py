import numpy
from PIL import Image
import torch
import os

_filedir = os.path.dirname(os.path.realpath(__file__))
_filename = 'motorbike.jpg'
_filepath = os.path.join(_filedir, _filename)

with Image.open(_filepath) as pil_img:
    img_motorbike = numpy.asarray(pil_img)

boxes_pred_motorbike = torch.tensor([[25, 12, 242, 215],  # 1
                                     [362, 126, 469, 425],  # 2
                                     [87, 241, 221, 343],  # 3
                                     [41, 52, 353, 283],  # 4
                                     [287, 76, 439, 388]], dtype=torch.float)  # 5

cls_pred_motorbike = torch.tensor([[0.2, 0.7, 0.1],
                                   [0.1, 0.05, 0.85],
                                   [0.1, 0.05, 0.85],
                                   [0.2, 0.7, 0.1],
                                   [0.1, 0.05, 0.85], ], dtype=torch.float)

boxes_target_motorbike = torch.tensor([[52, 22, 308, 328],  # A
                                       [26, 106, 456, 394]], dtype=torch.float)  # B

cls_target_motorbike = torch.tensor([0, 1], dtype=torch.long)

mask_target_motorbike = torch.tensor([True, True], dtype=torch.bool)
