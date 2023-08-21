import torch


def convert_target_to_dict(target):
    """Converts the annotations to a dict of tensors.
    :target: list of annotations
    :returns: dict of tensors with padded annotations
        "boxes": tensor of shape (B, N, 4)
        "labels": tensor of shape (B, N)
        "mask": tensor of shape (B, N) with 1 for valid annotations and 0 for padded annotations
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