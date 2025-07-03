"""
Author: liudec dec_hi@qq.com
Description: run rusboost classifier
"""

from imblearn.ensemble import RUSBoostClassifier
from utils.model_eval import model_eval
import numpy as np


def run_rusboost(x_train, x_test, y_train, y_test):
    """run rusboost

    Args:
        x_train, x_test, y_train, y_test: data and label
    """
    clf = RUSBoostClassifier(random_state=0)
    test_output_y = clf.fit(x_train, np.argmax(y_train,
                                               axis=1)).predict(x_test)
    acc, tnr, fpr, fnr, tpr = model_eval(test_output_y, y_test)
    print(f'accuracy {acc:.5f}, tpr {tpr:.5f}, '
          f'tnr {tnr:.5f}, fpr {fpr:.5f}, fnr {fnr:.5f}')
