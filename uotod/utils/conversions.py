import torch


def convert_target_to_dict(target):
    r"""Converts COCO annotations to a dict of tensors.

    :param target: list of annotations in the COCO format
    :type target: list[dict]
    :returns: dict of tensors with padded annotations
        "boxes": tensor of shape (B, N, 4)
        "labels": tensor of shape (B, N)
        "mask": tensor of shape (B, N) with 1 for valid annotations and 0 for padded annotations
        where B is the batch size and N is the maximum number of annotations in the batch.
    :rtype: dict[str, torch.Tensor]

    .. example::
        >>> target = [
        >>>     {
        >>>         "boxes": torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]),
        >>>         "labels": torch.tensor([1, 2]),
        >>>     },
        >>>     {
        >>>         "boxes": torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.], [0., 0., 1., 1.]]),
        >>>         "labels": torch.tensor([1, 2, 3]),
        >>>     },
        >>> ]
        >>> convert_target_to_dict(target)
        {'boxes': tensor([[[0., 0., 1., 1.],
                           [0., 0., 1., 1.],
                           [0., 0., 0., 0.]],
                          [[0., 0., 1., 1.],
                           [0., 0., 1., 1.],
                           [0., 0., 1., 1.]]]),
         'labels': tensor([[1, 2, 0],
                           [1, 2, 3]]),
         'mask': tensor([[ True,  True, False],
                         [ True,  True,  True]])
        }
    """

    device = target[0]['boxes'].device
    max_num_boxes = max([t['boxes'].shape[0] for t in target])
    num_images = len(target)

    boxes = torch.zeros(num_images, max_num_boxes, 4, dtype=torch.float, device=device)
    labels = torch.zeros(num_images, max_num_boxes, dtype=torch.long, device=device)
    mask = torch.zeros(num_images, max_num_boxes, dtype=torch.bool, device=device)

    for i, t in enumerate(target):
        num_boxes = t['boxes'].shape[0]
        boxes[i, :num_boxes, :] = t['boxes']
        labels[i, :num_boxes] = t['labels']
        mask[i, :num_boxes] = True

    return {"boxes": boxes, "labels": labels, "mask": mask}


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
