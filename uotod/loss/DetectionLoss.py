from typing import Union, Dict, List, Optional, Tuple

import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor

from ..match._Match import _Match
from .MultipleObjectiveLoss import MultipleObjectiveLoss
from ..utils import convert_target_to_dict


# TODO: write "class MultipleOutputsDetectionLoss(DetectionLoss)" which can handle multiple outputs (e.g. DETR)
# TODO: make compatible with DDP (e.g. to correctly average the losses ?! Note: his was not done in UOTOD's paper)
class DetectionLoss(_Loss):
    def __init__(self,
                 cls_loss_module: Union[MultipleObjectiveLoss, _Loss],
                 loc_loss_module: Union[MultipleObjectiveLoss, _Loss],
                 matching_method: _Match,
                 bg_class_position: str = "first",
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        r"""Detection loss.
        :param cls_loss_module: Classification loss used to compute the matching.
        :param loc_loss_module: Localization loss used to compute the matching.
        :param matching_method: Matching method used to compute the matching.
        :param bg_class_position: Index of the background class. "first", "last" or "none" (no background class).
        :param size_average: Deprecated.
        :param reduce: Deprecated.
        :param reduction: Type of reduction to apply to the final loss. "mean" or "sum".

        .. note::
            The final loss is the sum of the classification and localization losses, weighted by the matching.
            The background class is not taken into account in the localization loss.

        .. note::
            For a cross-entropy loss, use cls_annotation_format="index" and loss_cls_module=CrossEntropyLoss().
            For a binary cross-entropy loss, use cls_annotation_format="one-hot", bg_class_position="none" and
             loss_cls_module=BCEWithLogitsLoss().
            For a focal loss, use cls_annotation_format="one-hot", bg_class_position="none" and
             loss_cls_module=uotod.loss.FocalLoss().
        """
        super(DetectionLoss, self).__init__(size_average, reduce, reduction)

        assert isinstance(matching_method, _Match), "matching_method must be an instance of _Match"
        assert bg_class_position in ["first", "last", "none"], \
            "bg_class_index must be 'first', 'last' or 'none'"

        self.matching_method = matching_method
        self.loss_cls_module = cls_loss_module
        self.loss_loc_module = loc_loss_module
        self.bg_class_position = bg_class_position

    def forward(self,
                input: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                target: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
                anchors: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the matching between the predicted and target boxes, and the corresponding loss.
        :param input: dictionary containing the predicted logits and boxes.
            "pred_logits": Tensor of shape (batch_size, num_pred, num_classes).
            "pred_boxes": Tensor of shape (batch_size, num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :param target: dictionary containing the target classes, boxes and mask.
            "labels": Tensor of shape (batch_size, num_targets).
            "boxes": Tensor of shape (batch_size, num_targets, 4), where the last dimension is (x1, y1, x2, y2).
            "mask": Tensor of shape (batch_size, num_targets).
        :param anchors: the anchors used to compute the predicted boxes.
            Tensor of shape (batch_size, num_pred, 4) or (num_pred, 4), where the last dimension is (x1, y1, x2, y2).
        :return: the matching between the predicted and target boxes:
            Tensor of shape (batch_size, num_pred, num_targets).
        """
        # Convert the target to dict of masked tensors if necessary.
        if not isinstance(target, dict):
            target = convert_target_to_dict(target)

        if anchors.dim() == 2:
            anchors = anchors.unsqueeze(0).repeat(target['boxes'].shape[0], 1, 1)

        # Compute the matching.
        matching = self.matching_method(input, target, anchors)

        # Compute the loss terms.
        cls_loss = self._compute_cls_losses(input['pred_logits'], target['labels'])
        loc_loss = self._compute_loc_losses(input['pred_boxes'], target['boxes'])

        # Mask the loss matrices.
        tgt_mask = target['mask'].unsqueeze(dim=1).expand(loc_loss.shape)
        cls_loss[..., :-1] = cls_loss[..., :-1] * tgt_mask
        loc_loss = loc_loss * tgt_mask

        # Compute the total loss.
        # TODO: add option to use hard negative mining
        cls_loss = (matching * cls_loss)
        loc_loss = (matching[..., :-1] * loc_loss)

        # Average the loss terms.
        if self.reduction == 'mean':
            pos_weighting, neg_weighting = self._get_weighting(matching)
            cls_loss_reduced = cls_loss.sum() / (pos_weighting + neg_weighting)
            loc_loss_reduced = loc_loss.sum() / pos_weighting
            return cls_loss_reduced + loc_loss_reduced
        elif self.reduction == 'sum':
            cls_loss_reduced = cls_loss.sum()
            loc_loss_reduced = loc_loss.sum()
            return cls_loss_reduced + loc_loss_reduced
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

    def _get_weighting(self, matching) -> Tuple[Tensor, Tensor]:
        r"""
        Computes the weighting of the positive and negative samples in the loss.
        :param matching: the matching between the predicted and target boxes. Tensor of shape
            (batch_size, num_pred, num_targets).
        :return:
            - pos_weighting: the weighting of the positive samples. Tensor of shape (1,).
            - neg_weighting: the weighting of the negative samples. Tensor of shape (1,).

        .. note::
            The weighting is computed as the sum of the weights of the positive and negative samples.
        """
        if hasattr(self.loss_cls_module, "weight") and self.loss_cls_module.weight is not None and \
                self.bg_class_position == "first":
            pos_coef = self.loss_cls_module.weight[1:].mean()  # FIXME: incorrect (should use gather ?)
            neg_coef = self.loss_cls_module.weight[0]
        elif hasattr(self.loss_cls_module, "weight") and self.loss_cls_module.weight is not None and \
                self.bg_class_position == "last":
            pos_coef = self.loss_cls_module.weight[:-1].mean()  # FIXME: incorrect (should use gather ?)
            neg_coef = self.loss_cls_module.weight[-1]
        else:
            pos_coef = 1.
            neg_coef = 1.

        pos_weighting = matching[..., :-1].sum() * pos_coef
        if self.bg_class_position == "none":
            neg_weighting = 0
        else:
            neg_weighting = matching[..., -1].sum() * neg_coef

        return pos_weighting, neg_weighting

    def _compute_cls_losses(self, pred_logits: Tensor, tgt_labels: Tensor) -> Tensor:
        r"""
        Computes the classification cost matrix.
        :param pred_logits: Predicted logits. Tensor of shape (batch_size, num_pred, num_classes).
        :param tgt_labels: Target labels. Tensor of shape (batch_size, num_tgt) or
            (batch_size, num_targets, num_classes) if one-hot encoding is used.
        :return: Classification cost matrix. Tensor of shape (batch_size, num_pred, num_targets + 1).
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
            bg_class_index = num_classes - 1
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
                                         dtype=torch.int64, device=tgt_labels.device)
            tgt_classes_rep[..., :num_tgt] = tgt_labels.unsqueeze(dim=1).expand(batch_size, num_pred, num_tgt)
            tgt_classes_rep = tgt_classes_rep.view(batch_size * num_pred * (num_tgt + 1))

        # Compute the classification cost matrix.
        cls_loss = self.loss_cls_module(pred_logits_rep, tgt_classes_rep).view(batch_size, num_pred, num_tgt + 1)

        return cls_loss

    def _compute_loc_losses(self, pred_locs: Tensor, tgt_locs: Tensor) -> Tensor:
        r"""
        Computes the localization cost matrix.
        :param pred_locs: Predicted locations. Tensor of shape (batch_size, num_pred, 4).
        :param tgt_locs: Target locations. Tensor of shape (batch_size, num_targets, 4).
        :return: Localization cost matrix. Tensor of shape (batch_size, num_pred, num_targets).
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
            loc_loss = loc_loss.mean(dim=1)  # TODO: mean or sum reduction ?
        loc_loss = loc_loss.view(batch_size, num_pred, num_tgt)

        return loc_loss
