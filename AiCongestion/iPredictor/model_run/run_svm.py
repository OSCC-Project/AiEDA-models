"""
Author: liudec dec_hi@qq.com
Description: run svm model
"""

from sklearn.svm import SVC
from utils.model_eval import model_eval
import numpy as np


def run_svm(x_train, x_test, y_train, y_test):
    """run svm

    Args:
        x_train, x_test, y_train, y_test: data and label
    """
    kernels = ['poly', 'linear', 'rbf', 'sigmoid']
    for kernel in kernels:
        clf = SVC(kernel=kernel)
        test_output_y = clf.fit(x_train, np.argmax(y_train,
                                                   axis=1)).predict(x_test)
        acc, tnr, fpr, fnr, tpr = model_eval(test_output_y, y_test)
        print(f'kernel: {kernel}\t'
              f'accuracy {acc:.5f}, tpr {tpr:.5f}, '
              f'tnr {tnr:.5f}, fpr {fpr:.5f}, fnr {fnr:.5f}')
