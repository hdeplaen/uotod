import numpy
from PIL import Image
import torch

with Image.open('img.jpg') as pil_img:
    img = numpy.asarray(pil_img)


boxes_pred = torch.Tensor([[25, 12, 242, 215],      # 1
                           [362, 126, 469, 425],    # 2
                           [87, 241, 221, 343],     # 3
                           [41, 52, 353, 283],      # 4
                           [287, 76, 439, 388]])    # 5

boxes_target = torch.Tensor([[52, 22, 308, 328],    # A
                            [26, 106, 456, 394]])   # B

mask_pred = torch.BoolTensor([False, True, True, False, True])
mask_target = torch.BoolTensor([True, True])

input = {'pred_logits': None,
         'pred_boxes': boxes_pred.unsqueeze(0),
         'mask': mask_pred.unsqueeze(0)}
target = {'labels': None,
          'boxes': boxes_target.unsqueeze(0),
          'mask': mask_target.unsqueeze(0)}
