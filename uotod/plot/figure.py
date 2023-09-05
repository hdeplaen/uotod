from typing import Optional

from torch import Tensor, BoolTensor
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from .labels import target_labels, prediction_labels, target_colors, prediction_colors
from .params import ANNOTATE, ALPHA_BOXES, ALPHA_IMAGE, FONT_SIZE, TARGET_FILLED, PREDICTION_FILLED


def draw_box(ax, box, text=None, annotate=True, filled=False, color='black'):
    if filled:
        ax.add_patch(
            Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                      linewidth=0, facecolor=color, alpha=ALPHA_BOXES)
        )
    else:
        ax.add_patch(
            Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                      linewidth=1, facecolor='none', edgecolor=color)
        )
    if annotate:
        ax.annotate(text, (box[0] + 7, box[1] + 32),
                    color=color, fontsize=FONT_SIZE)


def draw_boxes(ax, boxes_pred, boxes_target, annotate=True,
               mask_pred: Optional[Tensor] = None, mask_target: Optional[Tensor] = None):
    if mask_pred is not None: pred = boxes_pred[mask_pred, :].cpu()
    else: pred = boxes_pred.cpu()

    if mask_target is not None: target = boxes_target[mask_target, :].cpu()
    else: target = boxes_target.cpu()

    num_pred = boxes_pred.size(0)
    num_target = boxes_target.size(0)

    for box, c, l in zip(target, target_colors(num_target+1, False, mask_target), target_labels(num_target+1, False, mask_target)):
        draw_box(ax, box, text=l, annotate=annotate, filled=TARGET_FILLED, color=c)
    for box, c, l in zip(pred, prediction_colors(num_pred, mask_pred), prediction_labels(num_pred, mask_pred)):
        draw_box(ax, box, l, annotate=annotate, filled=PREDICTION_FILLED, color=c)


def image_with_boxes(img, boxes_pred, boxes_target, mask_pred: Optional[BoolTensor] = None, mask_target: Optional[BoolTensor] = None):
    r"""
    :param img:  Image
    :param boxes_pred: Boxes of the predictions of size (number of boxes, coordinates). The coordinates are (top left x, top left y, width x, height y).
    :param boxes_target: Boxes of the ground truth objects of size (number of boxes, coordinates). The coordinates are (top left x, top left y, width x, height y).
    :param mask_pred: Mask of the predictions to be plotted.
    :param mask_target: Mask of the targets objects to be plotted.
    :type boxes_pred: Tensor
    :type boxes_target: Tensor
    :type mask_pred: BoolTensor, optional
    :type mask_target: BoolTensor, optional
    :return: Figure (matplotlib)
    """

    x_max, y_max, _ = img.shape

    fig = plt.figure(frameon=False)
    if isinstance(img, Tensor):
        plt.imshow(img.cpu())
    else:
        plt.imshow(img)

    ax = plt.gca()
    ax.set_axis_off()

    if ALPHA_IMAGE > 0.:
        ax.add_patch(Rectangle((0, 0), y_max, x_max, facecolor='#ffffff', linewidth=0, alpha=1 - ALPHA_IMAGE))

    draw_boxes(ax, boxes_pred, boxes_target, ANNOTATE, mask_pred, mask_target)

    fig.tight_layout(pad=0)
    return fig
