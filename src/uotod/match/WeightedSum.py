from typing import Union, Dict, Optional, Iterable, Tuple, List
from warnings import warn
from numpy import ndarray

import torch
from torch import Tensor

from .. import plot
from ..utils import kwargs_decorator
from ._Match import _Match

class WeightedSum(_Match):

    @kwargs_decorator({"matching_modules": None,
                       "weights": None,
                       "normalize": True,
                       "same_cost": True})
    def __init__(self, **kwargs) -> None:
        super(WeightedSum, self).__init__(**kwargs)
        self.matching_modules = kwargs["matching_modules"]
        self.weights = kwargs['weights']
        self.normalize = kwargs['normalize']
        self.same_cost = kwargs['same_cost']

    @property
    def matching_modules(self) -> List:
        return self._matching_modules

    @property
    def weights(self) -> List:
        return self._weights

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def same_cost(self) -> bool:
        return self._same_cost

    @matching_modules.setter
    def matching_modules(self, val: Iterable):
        assert val is not None, TypeError('The argument matching_modules is missing')
        assert isinstance(val, Iterable), TypeError('The matching_modules must be an iterable (a list typically).')
        self._matching_modules = list()
        for mod in val:
            assert isinstance(mod, _Match), TypeError('The modules in the matching_modules must all be of matching '
                                                      'type.')
            self._matching_modules.append(mod)
        self._num_modules = len(self._matching_modules)
        if self.background:
            for m in self._matching_modules:
                assert m.background, ValueError('The WeightedSum matching strategy requires a background, '
                                                'but some constituents do not contain one')

    @weights.setter
    def weights(self, val: Iterable):
        if val is None:
            self._weights = [1] * self._num_modules
        else:
            assert isinstance(self._weights, Iterable), TypeError(
                'If provided, the weights argument must be an iterable (a list typically).')
            self._weights = list()
            for w in val:
                self._weights.append(w)
            assert len(self._weights) == self._num_modules, ValueError(
                'The number of weights is not the same as the number of matching modules')
        self._inv_sum_weights = 1 / sum(self._weights)

    @normalize.setter
    def normalize(self, val: bool):
        assert isinstance(val, bool), TypeError('The normalize argument must be a boolean')
        self._normalize = val

    @same_cost.setter
    def same_cost(self, val: bool):
        assert isinstance(val, bool), TypeError('The same_cost argument must be a boolean')
        self._same_cost = val

    @torch.no_grad()
    def forward(self,
                input: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                target: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                anchors: Optional[Tensor] = None,
                cost_matrix: Optional[Tensor] = None,
                save: bool = True) \
            -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""
        Computes a batch of matchings between the predicted and target boxes.

        :param input: Input containing the predicted logits and boxes.
            "pred_logits": Tensor of shape (batch_size, num_pred, num_classes).
            "pred_boxes": Tensor of shape (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type input: dictionary
        :param target: Target containing the target classes, boxes and mask.
            "labels": Tensor of shape (batch_size, num_targets).
            "boxes": Tensor of shape (batch_size, num_targets, 4), where the last dimension is (x1, y1, x2, y2).
            "mask": Tensor of shape (batch_size, num_targets).
        :type target: dictionary
        :param anchors: the anchors used to compute the predicted boxes.
            (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type anchors: Tensor
        :return: the matching between the predicted and target boxes, and the cost matrix if returns_cost is True:
            Tensor of shape (batch_size, num_pred, num_targets + 1). The last entry of the last dimension is the
            background.
        :rtype: Tensor (float) or Tuple(Tensor, Tensor)

        """
        # emptying the cache
        self._last_cost = None
        self._last_match = None
        self._last_input = None
        self._last_target = None
        self._all_matchings = None

        matchings: List[Tensor] = list()
        costs: List[Tensor] = list()
        if self.same_cost:
            if cost_matrix is None:
                cost_matrix = self.compute_cost_matrix(input, target, anchors)
            for mod, w in zip(self.matching_modules, self.weights):
                loc_match = mod.compute_matching(cost_matrix, target["mask"])
                if mod.background and not self.background:
                    loc_match = loc_match[:,:,:-1]
                matchings.append(w * loc_match)
            if save:
                self._last_cost = cost_matrix
        else:
            for mod, w in zip(self.matching_modules, self.weights):
                loc_cost = mod.compute_cost_matrix(input, target, anchors)
                loc_match = mod.compute_matching(cost_matrix, target["mask"])
                if mod.background and not self.background:
                    loc_match = loc_match[:, :, :-1]
                if save: costs.append(loc_cost)
                matchings.append(w * loc_match)
            if save: self._last_cost = costs
        if save: self._all_matchings = matchings

        matching: Tensor = sum(matchings)
        if self.normalize:
            matching = matching * self._inv_sum_weights

        # Verify that the matching is not NaN.
        if torch.isnan(matching).any():
            raise RuntimeError("The matching is NaN, this may be caused by a too low regularization parameter.")

        # saving for the plotting
        if save:
            self._last_match = matching
            self._last_input = input
            self._last_target = target

        return matching

    def compute_matching(self, cost_matrix: Tensor, target_mask: Optional[Tensor]) -> Tensor:
        r"""
        Computes the matching.

        :param cost_matrix: Cost matrix of shape (batch_size, num_pred, num_targets + 1).
        :param target_mask: Target mask of shape (batch_size, num_targets).
        :type cost_matrix: Tensor
        :type target_mask: BoolTensor, optional
        :return: The matching :math:`\mathbf{P}` for each element of the batch. Tensor of shape (batch_size, num_pred, num_targets + 1). The last entry of the last
            dimension [:, :, num_target+1] is the background.
        """
        raise Exception('Not applicable')

    def _compute_matching_apart(self, cost_matrix: Tensor, out_view: Tensor, **kwargs) -> Tensor:
        pass

    def plot(self, idx=0, img: Optional[Union[Tensor, ndarray]] = None,
             plot_cost: bool = True,
             plot_match: bool = True,
             max_background_match: Union[float, int] = 1.,
             background: bool = True,
             erase: bool = False):
        r"""
        Plots from the last batch
        # TODO: extensive description

        :param idx: Index of the image to be plotted.
        :type idx: int, optional
        :param img: Image to be plotted. If it is not None, the boxes plot is computed.
        :type img: Tensor or ndarray, optional
        :param plot_cost: Plots the cost matrix between the predictions and the targets, including background.
        :type plot_cost: bool, optional
        :param plot_match: Plots the cost matrix between the predictions and the targets, including background.
        :type plot_match: bool, optional
        :param max_background_match: A threshold to only plot relevant matched predictions.
            The predictions are only plotted if the value matched to the background does not exceed
            `max_background_match`. Defaults to 1.
        :type max_background_match: float, optional
        :return: Matplotlib figures
        :rtype: Tuple(fig, fig, fig)
        """
        if plot_cost and not self._same_cost:
            warn('Cannot plot the cost as a different one was used for each strategy in the sum.')
            plot_cost = False

        return _Match.plot(self, idx, img, plot_cost, plot_match, max_background_match, background, erase)

    def plots_individual(self, idx=0,
                         plot_costs: bool = True,
                         plot_matches: bool = True,
                         background: bool = True,
                         erase: bool = True):
        exception = Exception(
            'The model has saved no values to plot. Insure that forward has been called at least once with the save argument set to True (by default).')

        assert self._last_match is not None, exception
        assert self._last_target is not None, exception
        assert self._last_cost is not None, exception
        assert self._last_input is not None, exception

        if not self.background:
            warn('Cannot plot the background class as the model does not contain one')
            background = False

        if plot_costs and self._same_cost:
            warn('Cannot plot all different costs as the same is used. Please use the method plot for this.')
            plot_costs = False

        target_mask = self._last_target["mask"][idx, :]

        subtitles: List[str] = list()
        for mod, w in zip(self.matching_modules, self.weights):
            subtitles.append(f"{mod.__class__.__name__}\n(weight: {w})")

        if plot_costs:
            costs: List[Tensor] = list()
            for c in self._last_cost:
                costs.append(c[idx, :, :])
            fig_costs = plot.multiple_costs(costs,
                                                  mask_pred=None,
                                                  mask_target=target_mask,
                                                  title='Costs in WeightedSum',
                                                  subtitles=subtitles,
                                                  background=background)
        else:
            fig_costs = None

        if plot_matches:
            matchings: List[Tensor] = list()
            for m in self._all_matchings:
                matchings.append(m[idx, :, :])
            fig_matches = plot.multiple_matches(matchings,
                                                  mask_pred=None,
                                                  mask_target=target_mask,
                                                  title='Matches in WeightedSum',
                                                  subtitles=subtitles,
                                                  background=background)
        else:
            fig_matches = None

        if erase:
            self._last_cost = None
            self._last_match = None
            self._last_input = None
            self._last_target = None

        return fig_costs, fig_matches


