from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops


"""
The code here has not been tested thoroughly, use at your own risk
"""


def focal_loss(labels, logits, gamma=2.0):
    r"""
    Multi-class focal loss implementation: https://arxiv.org/abs/1708.02002
    :param labels: [batch_size, ] - Tensor of the correct class ids
    :param logits: [batch_size, num_classes] - Unscaled logits
    :param gamma: focal loss weight
    :return: [batch_size, ] - Tensor of average costs for each batch element
    """

    num_classes = array_ops.shape(logits)[1]
    onehot_labels = array_ops.one_hot(labels, num_classes, dtype=logits.dtype)

    p = nn_ops.softmax(logits)
    p = clip_ops.clip_by_value(p, 1e-7, 1.0 - 1e-7)

    f_loss = - onehot_labels * math_ops.pow(1.0 - p, gamma) * math_ops.log(p) \
             - (1 - onehot_labels) * math_ops.pow(p, gamma) * math_ops.log(1.0 - p)

    cost = math_ops.reduce_sum(f_loss, axis=1)
    return cost


def mc_loss(labels, logits):
    r"""
    A multi-class cross-entropy loss
    :param labels: [batch_size, ] - Tensor of the correct class ids
    :param logits: [batch_size, num_classes] - Unscaled logits
    :return: [batch_size, ] - Tensor of average costs for each batch element
    """

    num_classes = array_ops.shape(logits)[1]
    onehot_labels = array_ops.one_hot(labels, num_classes, dtype=logits.dtype)

    p = nn_ops.softmax(logits)
    p = clip_ops.clip_by_value(p, 1e-7, 1.0 - 1e-7)

    ce_loss = - onehot_labels * math_ops.log(p) - (1 - onehot_labels) * math_ops.log(1.0-p)

    cost = math_ops.reduce_sum(ce_loss, axis=1)
    return cost
