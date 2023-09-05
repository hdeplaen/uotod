from matplotlib.colors import hsv_to_rgb
from string import ascii_uppercase
from typing import Optional
from torch import BoolTensor

from .params import PREDICTION_COLOR, TARGET_COLOR, PREDICTION_LABEL, TARGET_LABEL


# COLORS
def _color(num: int, color: str, mask: Optional[BoolTensor] = None):
    if color == 'cyclic':
        if mask is None:
            for i in range(num):
                yield hsv_to_rgb([(i * 0.3) % 1.0, 1, 1])
        else:
            for i in range(num):
                if mask[i]: yield hsv_to_rgb([(i * 0.3) % 1.0, 1, 1])
    elif isinstance(color, str):
        for _ in range(num):
            yield color
    else:
        raise Exception("The specified color must be a string.")


def target_colors(num_without_background, background: bool = False, mask: Optional[BoolTensor] = None):
    yield from _color(num_without_background, TARGET_COLOR, mask)
    if background:
        yield 'black'


def prediction_colors(num, mask: Optional[BoolTensor] = None):
    yield from _color(num, PREDICTION_COLOR, mask)


# LABELS
def _labels(num, type: str, mask: Optional[BoolTensor] = None):
    if type == 'numbers':
        if mask is None:
            yield from range(1, num)
        else:
            for i in range(num):
                if mask[i]: yield i+1
    elif type == 'letters':
        if mask is None:
            for i in range(num):
                yield ascii_uppercase[i % 26] * int(i / 26 + 1)
        else:
            for i in range(num):
                if mask[i]: yield ascii_uppercase[i % 26] * int(i / 26 + 1)
    else:
        raise Exception('The specified letters must be either "number" or "letters".')


def prediction_labels(num, mask: Optional[BoolTensor] = None):
    yield from _labels(num, PREDICTION_LABEL, mask)


def target_labels(num_without_background, background: bool = False, mask: Optional[BoolTensor] = None):
    yield from _labels(num_without_background , TARGET_LABEL, mask)
    if background: yield '$\\varnothing$'
