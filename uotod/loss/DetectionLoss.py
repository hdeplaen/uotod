import warnings
from typing import Union, Dict, List, Optional, Tuple

import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor

from ..match._Match import _Match
from .MultipleObjectiveLoss import MultipleObjectiveLoss
from ..utils import convert_target_to_dict


class DetectionLoss(_Loss):
    r""" Loss function for object detection.

    :param cls_loss_module: Classification loss.
    :type cls_loss_module: _Loss
    :param loc_loss_module: Localization loss.
    :type loc_loss_module: _Loss
    :param matching_method: Matching method used to compute the matching.
    :type matching_method: _Match
    :param bg_class_position: Index of the background class. "first", "last" or "none" (no background class).
    :type bg_class_position: str, optional
    :param use_hard_negative_mining: Whether to use hard negative mining.
    :type use_hard_negative_mining: bool, optional
    :param neg_to_pos_ratio: Ratio of negative to positive samples to use when using hard negative mining.
    :type neg_to_pos_ratio: float, optional
    :param size_average: Deprecated.
    :type size_average: bool, optional
    :param reduce: Deprecated.
    :type reduce: bool, optional
    :param reduction: Type of reduction to apply to the final loss. "mean" or "sum".
    :type reduction: str, optional
    :return: loss
    :rtype: Tensor (float)

    .. note::
        To use multiple localization or classification loss terms, use the class :class:`uotod.loss.MultipleObjectiveLoss`.

    .. note::
        The classification loss is averaged over the number of positive and negative samples, weighted by the
        class weights. This follows the implementation of the cross-entropy loss with class weights in PyTorch.

        Note that the number of negative samples is zero when using the focal loss, since there is no explicit
        background class.

        When using Hard Negative Mining, the number of negative samples is not taken into account in the averaging,
        for consistency with the SSD implementation.

    """

    def __init__(self,
                 cls_loss_module: Union[MultipleObjectiveLoss, _Loss],
                 loc_loss_module: Union[MultipleObjectiveLoss, _Loss],
                 matching_method: _Match,
                 bg_class_position: str = "first",
                 use_hard_negative_mining: bool = False,
                 neg_to_pos_ratio: Optional[float] = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super(DetectionLoss, self).__init__(size_average, reduce, reduction)

        assert isinstance(matching_method, _Match), "matching_method must be an instance of _Match"
        assert bg_class_position in ["first", "last", "none"], \
            "bg_class_position must be 'first', 'last' or 'none'"

        self.matching_method = matching_method
        self.loss_cls_module = cls_loss_module
        self.loss_loc_module = loc_loss_module
        self.bg_class_position = bg_class_position

        self.use_hard_negative_mining = use_hard_negative_mining
        if self.use_hard_negative_mining:
            assert neg_to_pos_ratio is not None, "neg_to_pos_ratio must be specified when using hard negative mining"
        elif neg_to_pos_ratio is not None:
            warnings.warn("neg_to_pos_ratio is ignored when not using hard negative mining")
        self.neg_to_pos_ratio = neg_to_pos_ratio

    def forward(self,
                input: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                target: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                anchors: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the matching between the predicted and target boxes, and the corresponding loss.

        :param input: Input containing the predicted logits and boxes.

            "pred_logits": Tensor of shape (batch_size, num_pred, num_classes).

            "pred_boxes": Tensor of shape (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).

        :type input: dictionary
        :param target: Target containing the target classes, boxes and mask.

            "labels": Ground-truth labels. Tensor of shape (batch_size, num_targets).

            "boxes": Ground-truth bounding boxes. Tensor of shape (batch_size, num_targets, 4), where the last dimension is (x1, y1, x2, y2).

            "mask": Padding mask. Tensor of shape (batch_size, num_targets).

        :type target: dictionary
        :param anchors: the anchors used to compute the predicted boxes, optional.
            (batch_size, num_pred, 4) or (num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :type anchors: Tensor
        :return: loss
        :rtype: Tensor (float)

        Examples::

            >>> import torch
            >>> from uotod.loss import DetectionLoss, NegativeProbLoss, GIoULoss, IoULoss
            >>> from uotod.match import Hungarian, UnbalancedSinkhorn
            >>> from uotod.utils import box_cxcywh_to_xyxy
            >>> # Define the matching method
            >>> matching_method = Hungarian(
            >>>     cls_match_module=NegativeProbLoss(reduction="none"),
            >>>     loc_match_module=torch.nn.L1Loss(reduction="none")
            >>> )
            >>> # Define the loss function
            >>> loss_fn = DetectionLoss(
            >>>     cls_loss_module=torch.nn.CrossEntropyLoss(reduction="none"),
            >>>     loc_loss_module=GIoULoss(reduction="none"),
            >>>     matching_method=matching_method
            >>> )
            >>> # Define the input
            >>> pred = {"pred_logits": torch.randn(2, 100, 21),
            >>>         "pred_boxes": box_cxcywh_to_xyxy(torch.rand(2, 100, 4))}
            >>> # Define the target
            >>> target = {"labels": torch.randint(1, 21, (2, 10)),      # 0 is the background class
            >>>           "boxes": box_cxcywh_to_xyxy(torch.rand(2, 10, 4)),                # (x1, y1, x2, y2)
            >>>           "mask": torch.ones(2, 10, dtype=torch.bool)}  # Padding mask
            >>> # Compute the loss
            >>> loss_fn(pred, target)

            >>> # With anchors
            >>> anchors = box_cxcywh_to_xyxy(torch.rand(100, 4))
            >>> # Define the matching method
            >>> matching_method = Hungarian(
            >>>     cls_match_module=None,                                # No classification cost for matching anchors
            >>>     loc_match_module=IoULoss(reduction="none"),
            >>>     is_anchor_based=True                                # Use anchor-based matching
            >>> )
            >>> # Define the loss function
            >>> loss_fn = DetectionLoss(
            >>>     cls_loss_module=torch.nn.CrossEntropyLoss(reduction="none"),
            >>>     loc_loss_module=GIoULoss(reduction="none"),
            >>>     matching_method=matching_method
            >>> )
            >>> # Compute the loss
            >>> loss_fn(pred, target, anchors)

            >>> # Using Unbalanced Optimal Transport and hard negative mining
            >>> # Define the matching method
            >>> matching_method = UnbalancedSinkhorn(
            >>>     cls_match_module=None,
            >>>     loc_match_module=IoULoss(reduction="none"),
            >>>     reg_pred=1e+3,                                      # Regularization parameter for the predicted boxes
            >>>     reg_target=1e-3,                                    # Regularization parameter for the target boxes
            >>>     background_cost=0.5,                                # Threshold (IoU) for matching to background
            >>>     is_anchor_based=True                                # Use anchor-based matching
            >>> )
            >>> # Define the loss function
            >>> loss_fn = DetectionLoss(
            >>>     cls_loss_module=torch.nn.CrossEntropyLoss(reduction="none"),
            >>>     loc_loss_module=GIoULoss(reduction="none"),
            >>>     matching_method=matching_method,
            >>>     use_hard_negative_mining=True,                      # Use hard negative mining
            >>>     neg_to_pos_ratio=3.                                 # Use 3 negative samples for 1 positive sample
            >>> )
            >>> # Compute the loss
            >>> loss_fn(pred, target, anchors)
        """
        # Convert the target to dict of masked tensors, if necessary.
        if not isinstance(target, dict):
            target = convert_target_to_dict(target)

        # Repeat the anchors to match the batch size, if necessary.
        if anchors is not None and anchors.dim() == 2:
            anchors = anchors.unsqueeze(0).repeat(target['boxes'].shape[0], 1, 1)

        # Compute the matching.
        matching = self.matching_method(input, target, anchors)  # (batch_size, num_pred, num_targets + 1)

        # Compute the loss terms.
        cls_loss = self._compute_cls_losses(input['pred_logits'], target['labels'])  # (batch_size, num_pred, num_targets + 1)
        loc_loss = self._compute_loc_losses(input['pred_boxes'], target['boxes'])  # (batch_size, num_pred, num_targets)

        # Mask the loss matrices with the padding mask.
        tgt_mask = target['mask'].unsqueeze(dim=1).expand(loc_loss.shape)
        cls_loss[..., :-1] = cls_loss[..., :-1] * tgt_mask
        loc_loss = loc_loss * tgt_mask

        # Compute the total loss.
        cls_loss_pos = (matching[..., :-1] * cls_loss[..., :-1])
        cls_loss_neg = (matching[..., -1] * cls_loss[..., -1])
        loc_loss = (matching[..., :-1] * loc_loss)

        # Hard negative mining.
        if self.use_hard_negative_mining:
            cls_loss_neg = self._hard_negative_mining(cls_loss_neg, matching)

        # Average the loss terms.
        if self.reduction == 'mean':
            num_pos_weighted, num_neg_weighted = self._get_averaging_coefs(matching, target['labels'])
            if self.use_hard_negative_mining:
                cls_loss_reduced = (cls_loss_pos.sum() + cls_loss_neg.sum()) / num_pos_weighted
            else:
                cls_loss_reduced = (cls_loss_pos.sum() + cls_loss_neg.sum()) / (num_pos_weighted + num_neg_weighted)
            loc_loss_reduced = loc_loss.sum() / num_pos_weighted
            return cls_loss_reduced + loc_loss_reduced
        elif self.reduction == 'sum':
            cls_loss_reduced = (cls_loss_pos.sum() + cls_loss_neg.sum())
            loc_loss_reduced = loc_loss.sum()
            return cls_loss_reduced + loc_loss_reduced
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

    def _get_averaging_coefs(self, matching: Tensor, tgt_labels: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Computes the number of positive and negative matches, weighted by the class weights.

        :param matching: the matching between the predicted and target boxes. Tensor of shape
            (batch_size, num_pred, num_targets).
        :type matching: Tensor
        :param tgt_labels: Ground-truth labels. Tensor of shape (batch_size, num_tgt) or
            (batch_size, num_targets, num_classes) if one-hot encoding is used.
        :type tgt_labels: Tensor
        :return:
            - num_pos_weighted: the number of positive samples, weighted by the class weights.
            - num_neg_weighted: the number of negative samples, weighted by the background class weight.
        :rtype: tuple of Tensors
        """
        # FIXME: currently not compatible torch.nn.BCELoss() / torch.nn.BCELossWithLogits() weight parameter
        # Compute the weighting of the positive and negative classes.
        if hasattr(self.loss_cls_module, "weight") and self.loss_cls_module.weight is not None and \
                self.bg_class_position == "first":
            pos_coefs = self.loss_cls_module.weight[tgt_labels]  # (batch_size, num_targets)
            # old : pos_coefs = self.loss_cls_module.weight[1:][tgt_labels]
            neg_coef = self.loss_cls_module.weight[0]
        elif hasattr(self.loss_cls_module, "weight") and self.loss_cls_module.weight is not None and \
                self.bg_class_position == "last":
            pos_coefs = self.loss_cls_module.weight[tgt_labels]  # (batch_size, num_targets)
            # old : pos_coefs = self.loss_cls_module.weight[:-1][tgt_labels]
            neg_coef = self.loss_cls_module.weight[-1]
        else:
            pos_coefs = 1.
            neg_coef = 1.

        # Compute the number of positive samples: sum of the matching matrix, weighted by the gt class weights.
        num_pos_weighted = matching[..., :-1].sum(dim=1) * pos_coefs  # (batch_size, num_targets)
        num_pos_weighted = num_pos_weighted.sum()  # (1,)

        # Compute the number of negative samples.
        if self.bg_class_position == "none":
            num_neg_weighted = 0.
        else:
            num_neg_weighted = matching[..., -1].sum() * neg_coef  # (1,)

        # Clamp the number of positive samples to 1, to avoid division by 0.
        num_pos_weighted = torch.clamp(num_pos_weighted, min=1.)

        return num_pos_weighted, num_neg_weighted

    def _hard_negative_mining(self, cls_loss_neg: Tensor, matching: Tensor) -> Tensor:
        r"""
        Performs hard negative mining.

        :param cls_loss_neg: Classification loss for the negative samples. Tensor of shape (batch_size, num_pred).
        :param matching: the matching between the predicted and target boxes. Tensor of shape
            (batch_size, num_pred, num_targets+1).
        :return: Classification loss for the negative samples after hard negative mining.
        """

        # Compute the maximum number of negative samples to keep.
        max_num_negative = self.neg_to_pos_ratio * matching[..., :-1].sum(dim=(1, 2))

        # Sort the negative samples by decreasing loss.
        _, idx = cls_loss_neg.sort(1, descending=True)
        matching_bg_sorted = torch.gather(matching[:, :, -1], dim=1, index=idx)

        # Keep the max_num_negative negative samples with the highest loss.
        cumulative_num_negative = torch.cumsum(matching_bg_sorted, dim=1)
        background_idxs_unsorted = cumulative_num_negative < max_num_negative[:, None]
        background_idxs = torch.gather(background_idxs_unsorted, dim=1, index=idx.sort(1)[1])
        cls_loss_neg = cls_loss_neg[background_idxs]

        return cls_loss_neg

    def _compute_cls_losses(self, pred_logits: Tensor, tgt_labels: Tensor) -> Tensor:
        r"""
        Computes the classification cost matrix.
        :param pred_logits: Predicted logits. Tensor of shape (batch_size, num_pred, num_classes).
        :type pred_logits: Tensor
        :param tgt_labels: Ground-truth labels. Tensor of shape (batch_size, num_tgt) or
            (batch_size, num_targets, num_classes) if one-hot encoding is used.
        :type tgt_labels: Tensor
        :return: Classification cost matrix. Tensor of shape (batch_size, num_pred, num_targets + 1).
        :rtype: Tensor
        """
        batch_size, num_pred, num_classes = pred_logits.shape
        num_tgt = tgt_labels.shape[1]
        is_onehot = (tgt_labels.dim() == 3)

        if self.bg_class_position == "first":
            if is_onehot:
                raise NotImplementedError("bg_class_position='first' is not supported with one-hot encoding")
            bg_class_index = 0
        elif self.bg_class_position == "last":
            if is_onehot:
                raise NotImplementedError("bg_class_position='last' is not supported with one-hot encoding")
            bg_class_index = num_classes - 1  # num_classes is the number of classes, including the background class
        else:  # self.bg_class_position == "none"  (for focal loss)
            if not is_onehot:
                raise NotImplementedError("bg_class_position='none' is only supported with one-hot encoding")
            assert num_classes == tgt_labels.shape[2], \
                "num_classes must be equal to the number of classes in the one-hot encoding"
            bg_class_index = None

        pred_logits_rep = pred_logits.unsqueeze(dim=2).repeat(1, 1, num_tgt + 1, 1).view(
            batch_size * num_pred * (num_tgt + 1), -1)

        if is_onehot:
            tgt_classes_rep = tgt_labels.unsqueeze(dim=1).repeat(1, num_pred, 1, 1)
            tgt_classes_rep = torch.cat([tgt_classes_rep, torch.zeros_like(tgt_classes_rep[:, :, :1, :])], dim=2)
            tgt_classes_rep = tgt_classes_rep.view(batch_size * num_pred * (num_tgt + 1), num_classes)
        else:
            tgt_classes_rep = torch.full((batch_size, num_pred, num_tgt + 1), fill_value=bg_class_index,
                                         dtype=torch.long, device=tgt_labels.device)
            tgt_classes_rep[..., :num_tgt] = tgt_labels.unsqueeze(dim=1).expand(batch_size, num_pred, num_tgt)
            tgt_classes_rep = tgt_classes_rep.view(batch_size * num_pred * (num_tgt + 1))

        # Compute the classification cost matrix.
        cls_loss = self.loss_cls_module(pred_logits_rep, tgt_classes_rep).view(batch_size, num_pred, num_tgt + 1)

        return cls_loss

    def _compute_loc_losses(self, pred_locs: Tensor, tgt_locs: Tensor) -> Tensor:
        r"""
        Computes the localization cost matrix.
        :param pred_locs: Predicted locations. Tensor of shape (batch_size, num_pred, 4).
        :type pred_locs: Tensor
        :param tgt_locs: Ground-truth locations. Tensor of shape (batch_size, num_targets, 4).
        :type tgt_locs: Tensor
        :return: Localization cost matrix. Tensor of shape (batch_size, num_pred, num_targets).
        :rtype: Tensor
        """
        batch_size, num_pred = pred_locs.shape[:2]
        num_tgt = tgt_locs.shape[1]

        pred_locs_rep = pred_locs.unsqueeze(dim=2).repeat(1, 1, num_tgt, 1).view(
            batch_size * num_pred * num_tgt, 4)
        tgt_locs_rep = tgt_locs.unsqueeze(dim=1).repeat(1, num_pred, 1, 1).view(
            batch_size * num_pred * num_tgt, 4)

        # Compute the localization cost matrix.
        loc_loss = self.loss_loc_module(pred_locs_rep, tgt_locs_rep)
        if loc_loss.dim() == 2:
            loc_loss = loc_loss.sum(dim=1)
        loc_loss = loc_loss.view(batch_size, num_pred, num_tgt)

        return loc_loss
