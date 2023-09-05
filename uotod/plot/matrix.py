from typing import Optional
from torch import Tensor, BoolTensor, cat
from matplotlib import pyplot as plt

from .labels import prediction_labels, target_labels, prediction_colors, target_colors
from .params import MATCH_CMAP, COST_CMAP


def prune_matrix(m,
                 mask_pred: Optional[BoolTensor] = None,
                 mask_target: Optional[BoolTensor] = None,
                 background: bool = True):
    if mask_pred is not None:
        m = m[mask_pred, :]
    if mask_target is not None:
        if background:
            m = cat((m[:,:-1][:, mask_target],m[:,-1:]), dim=1)
        else:
            m = m[:,:-1][:, mask_target]
    return m


def matrix(m, cmap, mask_pred: Optional[BoolTensor] = None, mask_target: Optional[BoolTensor] = None, background: bool = True):
    num_pred, num_target = m.size()
    m = prune_matrix(m, mask_pred, mask_target, background)
    num_pred_pruned, num_target_pruned = m.size()
    if isinstance(m, Tensor): m = m.cpu()

    fig = plt.figure(frameon=False)
    plt.imshow(m, cmap=cmap)
    ax = plt.gca()

    pl = prediction_labels(num_pred, mask_pred)
    tl = target_labels(num_target-1, background, mask_target)
    pc = prediction_colors(num_pred, mask_pred)
    tc = target_colors(num_target-1, background, mask_target)

    ax.set_yticks(range(num_pred_pruned), pl)
    ax.set_xticks(range(num_target_pruned), tl)
    ax.tick_params(axis='both', which='both', length=0)

    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), pc):
        ticklabel.set_color(tickcolor)
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), tc):
        ticklabel.set_color(tickcolor)

    ax.set_xlabel('Targets')
    ax.set_ylabel('Predictions')
    plt.colorbar()
    return fig


def match(match,
          mask_pred: Optional[BoolTensor] = None,
          mask_target: Optional[BoolTensor] = None,
          title: str = 'Match',
          background: bool = True):
    r"""
    Generates the figure of the match as a matrix.

    :param match: Match to be plotted, of size (predictions, ground truth objects + background).
    :param mask_pred: Mask of the predictions to be plotted.
    :param mask_target: Mask of the ground truth objects to be plotted.
    :param title: Specific title to the figure.
    :param background: Specifies if the background is to be plotted.
    :type match: Tensor
    :type mask_pred: BoolTensor, optional
    :type mask_target: BoolTensor, optional
    :type title: str, optional
    :type background: bool, optional
    :return: Figure (matplotlib)
    """
    fig = matrix(match, MATCH_CMAP, mask_pred, mask_target, background)
    fig.suptitle(title)
    return fig


def cost(cost,
         mask_pred: Optional[BoolTensor] = None,
         mask_target: Optional[BoolTensor] = None,
         title: str = 'Cost',
         background: bool = True):
    r"""
    Generates the figure of the match of the cost as a matrix.

    :param match: Cost to be plotted, of size (predictions, ground truth objects + background).
    :param mask_pred: Mask of the predictions to be plotted.
    :param mask_target: Mask of the ground truth objects to be plotted.
    :param title: Specific title to the figure.
    :param background: Specifies if the background is to be plotted.
    :type match: Tensor
    :type mask_pred: BoolTensor, optional
    :type mask_target: BoolTensor, optional
    :type title: str, optional
    :type background: bool, optional
    :return: Figure (matplotlib)
    """

    fig = matrix(cost, COST_CMAP, mask_pred, mask_target, background)
    fig.suptitle(title)
    return fig
