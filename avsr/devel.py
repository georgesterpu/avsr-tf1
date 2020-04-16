from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow import losses as losses, dtypes

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


def smoothed_cross_entropy(num_classes, label_smoothing):
    def _smoothed_cross_entropy(labels=None, logits=None,
                                _label_smoothing=label_smoothing):
        onehot_labels = array_ops.one_hot(labels, num_classes,
                                          dtype=logits.dtype)
        return losses.softmax_cross_entropy(onehot_labels, logits,
                                            label_smoothing=_label_smoothing)
    return _smoothed_cross_entropy


def sentence_loss(labels, logits):
    argmax_labels = math_ops.argmax_v2(logits, axis=1, output_type=labels.dtype)
    perfect_match = math_ops.reduce_all(math_ops.equal(argmax_labels, labels))
    loss = 1.0 - math_ops.cast(perfect_match, dtype=logits.dtype)
    return loss
