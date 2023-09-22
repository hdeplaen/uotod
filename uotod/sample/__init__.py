import torch
from .motorbike import img_motorbike, boxes_pred_motorbike, boxes_target_motorbike, cls_pred_motorbike, \
    cls_target_motorbike, mask_target_motorbike
from .elephants import img_elephants, boxes_pred_elephants, boxes_target_elephants, cls_pred_elephants, \
    cls_target_elephants, mask_target_elephants

imgs = [img_motorbike, img_elephants]

input = {
    'pred_logits': torch.cat([cls_pred_motorbike.unsqueeze(0),
                              cls_pred_elephants.unsqueeze(0)],
                             dim=0),
    'pred_boxes': torch.cat([boxes_pred_motorbike.unsqueeze(0),
                             boxes_pred_elephants.unsqueeze(0)],
                            dim=0)
}
target = {
    'labels': torch.cat([cls_target_motorbike.unsqueeze(0),
                         cls_target_elephants.unsqueeze(0)],
                        dim=0),
    'boxes': torch.cat([boxes_target_motorbike.unsqueeze(0),
                        boxes_target_elephants.unsqueeze(0)],
                       dim=0),
    'mask': torch.cat([mask_target_motorbike.unsqueeze(0),
                       mask_target_elephants.unsqueeze(0)],
                      dim=0),
}

anchors = torch.tensor([[0, 0, 200, 200],  # 1
                        [300, 100, 400, 400],  # 2
                        [100, 250, 200, 320],  # 3
                        [50, 50, 350, 300],  # 4
                        [300, 50, 450, 400]], dtype=torch.float)  # 5

# move to cuda
if torch.cuda.is_available():
    input = {k: v.to('cuda') for k, v in input.items()}
    target = {k: v.to('cuda') for k, v in target.items()}
    anchors = anchors.to('cuda')
