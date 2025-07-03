"""
Author: liudec dec_hi@qq.com
Description: eval model
"""

import torch
import numpy as np


def model_eval(output, target):
    """model eval

    Args:
        output: tensor, its shape is [n], all data is 0 or 1
        target: tensor, its shape is [n], all data is 0 or 1

    Returns:
        acc, tnr, fpr, fnr, tpr: some model evaluation criteria
    """
    # change type into tensor if type is ndarray
    if type(output) == np.ndarray and type(target) == np.ndarray:
        output = torch.from_numpy(output)
        target = torch.from_numpy(target)

    # compute num
    confusion_vector = output / target
    tp_num = torch.sum(confusion_vector == 1).item()
    fp_num = torch.sum(confusion_vector == float('inf')).item()
    tn_num = torch.sum(torch.isnan(confusion_vector)).item()
    fn_num = torch.sum(confusion_vector == 0).item()
    correct = (output == target).float().sum()

    # compute the result
    acc = 100 * correct / output.shape[0]
    tnr = 100 * tn_num / (tn_num + fp_num) if (tn_num + fp_num) != 0 else 0
    fpr = 100 * fp_num / (fp_num + tn_num) if (fp_num + tn_num) != 0 else 0
    fnr = 100 * fn_num / (tp_num + fn_num) if (tp_num + fn_num) != 0 else 0
    tpr = 100 * tp_num / (tp_num + fn_num) if (tp_num + fn_num) != 0 else 0
    return acc, tnr, fpr, fnr, tpr
