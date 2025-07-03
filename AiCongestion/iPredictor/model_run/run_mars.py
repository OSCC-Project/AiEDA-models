"""
Author: liudec dec_hi@qq.com
Description: run multivariate adaptive regression splines model (MARS)
"""

from pyearth import Earth
from utils.model_eval import model_eval
import numpy as np


def run_mars(x_train, x_test, y_train, y_test):
    """run mars: multivariate adaptive regression splines model

    Args:
        x_train, x_test, y_train, y_test: data and label
    """
    np.warnings.filterwarnings('ignore')
    clf = Earth()
    test_output_y = clf.fit(x_train, np.argmax(y_train,
                                               axis=1)).predict(x_test)
    test_output_y = test_output_y > 0.5
    acc, tnr, fpr, fnr, tpr = model_eval(test_output_y, y_test)
    print(f'accuracy {acc:.5f}, tpr {tpr:.5f}, '
          f'tnr {tnr:.5f}, fpr {fpr:.5f}, fnr {fnr:.5f}')
