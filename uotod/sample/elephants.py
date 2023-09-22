import numpy
from PIL import Image
import torch
import os

_filedir = os.path.dirname(os.path.realpath(__file__))
_filename = 'elephants.jpg'
_filepath = os.path.join(_filedir, _filename)

with Image.open(_filepath) as pil_img:
    img_elephants = numpy.asarray(pil_img)

boxes_pred_elephants = torch.tensor([[130., 160., 320., 320.],
                                     [350., 280., 550., 400.],
                                     [105., 130., 265., 260.],
                                     [300., 105., 570., 295.],
                                     [20., 50., 120., 120.]])

cls_pred_elephants = torch.tensor([[0.2, 0.7, 0.1],
                                   [0.1, 0.05, 0.85],
                                   [0.1, 0.05, 0.85],
                                   [0.2, 0.7, 0.1],
                                   [0.1, 0.05, 0.85], ], dtype=torch.float)

boxes_target_elephants = torch.tensor([[292.62, 134.59, 518.84, 285.05],
                                       [114.35, 148.97, 341.14, 297.63]])

cls_target_elephants = torch.tensor([0, 1], dtype=torch.long)

mask_target_elephants = torch.tensor([True, True], dtype=torch.bool)
