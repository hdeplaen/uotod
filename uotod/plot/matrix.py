from typing import Optional, Tuple, List
from torch import Tensor, BoolTensor, cat
from matplotlib import pyplot as plt
import torch

from .labels import prediction_labels, target_labels, prediction_colors, target_colors
from .params import MATCH_CMAP, COST_CMAP


def prune_matrix(m,
                 mask_pred: Optional[BoolTensor] = None,
                 mask_target: Optional[BoolTensor] = None,
                 background: bool = True) -> Tuple[Tensor, int, int]:
    num_pred = m.size(0)
    if not background and mask_target.size(0) == m.size(1):
        num_target = m.size(1)
    else:
        num_target = m.size(1) - 1

    if mask_pred is not None:
        m = m[mask_pred, :]
    if mask_target is not None:
        if background:  # if a background is specified
            m = cat((m[:, :-1][:, mask_target], m[:, -1:]), dim=1)
        elif not background and mask_target.size(0) == m.size(1):  # if the background does not exist
            m = m[:, mask_target]
        else:  # if the background exists and is to be removed
            m = m[:, :-1][:, mask_target]

    return m, num_pred, num_target


def matrix(ax, m, cmap, vmin, vmax,
           mask_pred: Optional[BoolTensor] = None,
           mask_target: Optional[BoolTensor] = None,
           background: bool = True,
           subtitle: Optional[str] = None,
           prediction_label=True,
           target_label=False):
    m, num_pred, num_target = prune_matrix(m, mask_pred, mask_target, background)
    num_pred_pruned, num_target_pruned = m.size()
    if isinstance(m, Tensor): m = m.cpu()

    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    pl = prediction_labels(num_pred, mask_pred)
    tl = target_labels(num_target, background, mask_target)
    pc = prediction_colors(num_pred, mask_pred)
    tc = target_colors(num_target, background, mask_target)

    ax.set_yticks(range(num_pred_pruned), pl)
    ax.set_xticks(range(num_target_pruned), tl)
    ax.tick_params(axis='both', which='both', length=0)

    for ticklabel, tickcolor in zip(ax.get_yticklabels(), pc):
        ticklabel.set_color(tickcolor)
    for ticklabel, tickcolor in zip(ax.get_xticklabels(), tc):
        ticklabel.set_color(tickcolor)

    if prediction_label: ax.set_ylabel('Predictions')
    if target_label: ax.set_xlabel('Targets')
    if subtitle is not None: ax.set_title(subtitle)
    return im


def _multiple_matrices(ms: list,
                       cmap,
                       mask_pred: Optional[BoolTensor] = None,
                       mask_target: Optional[BoolTensor] = None,
                       title: Optional[str] = 'Match',
                       subtitles: Optional[List[str]] = None,
                       background: bool = True,
                       subplots_disp: Optional[Tuple[int, int]] = None):
    num = len(ms)
    assert num > 0, 'No example provided to plot.'
    if subplots_disp is not None:
        nrows, ncols = subplots_disp
        assert nrows * ncols == num, 'The subplots disposition does not correspond to the number of provided examples.'
    else:
        nrows, ncols = 1, num

    vmin, vmax = torch.inf, -torch.inf
    for m in ms:
        m_min, m_max = m.min(), m.max()
        if m_min < vmin: vmin = m_min
        if m_max > vmax: vmax = m_max

    if subtitles is None:
        subtitles = [None] * num

    fig, axs = plt.subplots(nrows, ncols, frameon=False, squeeze=False)
    for idx in range(num):
        im = matrix(axs[idx//ncols, idx % ncols],
                    ms[idx], cmap, float(vmin), float(vmax),
                    mask_pred, mask_target, background, subtitles[idx],
                    idx % ncols == 0, idx // ncols == nrows-1)
    fig.colorbar(im, ax=axs.ravel().tolist(), cmap=cmap)
    if title is not None:
        fig.suptitle(title)
    return fig


def multiple_matches(matches: list,
                     mask_pred: Optional[List[BoolTensor]] = None,
                     mask_target: Optional[List[BoolTensor]] = None,
                     title: Optional[str] = 'Matches',
                     subtitles: Optional[List[str]] = None,
                     background: bool = True,
                     subplots_disp: Optional[Tuple[int, int]] = None):
    r"""
    Stacks multiple results of the function match next to each other.

    :param matches: Matches to be plotted as a list of Tensors of size (predictions, ground truth objects + background).
    :param mask_pred: Mask of the predictions to be plotted.
    :param mask_target: Mask of the ground truth objects to be plotted.
    :param title: Specific title to the figure.
    :param subtitles: Specific subtitles for each subplot.
    :param background: Specifies if the background is to be plotted.
    :param subplots_disp: Disposition of the subplots. Defaults to (1, len(matches)).
    :type matches: List[Tensor]
    :type mask_pred: BoolTensor, optional
    :type mask_target: BoolTensor, optional
    :type title: str, optional
    :type subtitles: List[str], optional
    :type background: bool, optional
    :type subplots_disp: Tuple[int, int], optional
    :return: Figure (matplotlib)
    """
    return _multiple_matrices(matches,
                              MATCH_CMAP,
                              mask_pred,
                              mask_target,
                              title, subtitles,
                              background,
                              subplots_disp)


def multiple_costs(costs: list,
                   mask_pred: Optional[List[BoolTensor]] = None,
                   mask_target: Optional[List[BoolTensor]] = None,
                   title: Optional[str] = 'Costs',
                   subtitles: Optional[List[str]] = None,
                   background: bool = True,
                   subplots_disp: Optional[Tuple[int, int]] = None):
    r"""
    Stacks multiple results of the function cost next to each other.

    :param costs: Costs to be plotted, as a list of Tensors of size (predictions, ground truth objects + background).
    :param mask_pred: Mask of the predictions to be plotted.
    :param mask_target: Mask of the ground truth objects to be plotted.
    :param title: Specific title to the figure.
    :param subtitles: Specific subtitles for each subplot.
    :param background: Specifies if the background is to be plotted.
    :param subplots_disp: Disposition of the subplots. Defaults to (1, len(matches)).
    :type costs: List[Tensor]
    :type mask_pred: BoolTensor, optional
    :type mask_target: BoolTensor, optional
    :type title: str, optional
    :type subtitles: List[str], optional
    :type background: bool, optional
    :type subplots_disp: Tuple[int, int], optional
    :return: Figure (matplotlib)
    """
    return _multiple_matrices(costs,
                              COST_CMAP,
                              mask_pred,
                              mask_target,
                              title, subtitles,
                              background,
                              subplots_disp)


def cost(cost,
         mask_pred: Optional[BoolTensor] = None,
         mask_target: Optional[BoolTensor] = None,
         title: Optional[str] = 'Cost',
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
    return multiple_costs([cost], mask_pred, mask_target, title, None, background, (1,1))


def match(match,
          mask_pred: Optional[BoolTensor] = None,
          mask_target: Optional[BoolTensor] = None,
          title: Optional[str] = 'Cost',
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
    return multiple_matches([match], mask_pred, mask_target, title, None, background, (1,1))
