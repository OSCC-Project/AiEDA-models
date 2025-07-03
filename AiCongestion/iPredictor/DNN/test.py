"""
Author: juanyu 291701755@qq.com
Description: train DNN model
"""

import torch
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

from DNN import DNN

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
    TP = torch.sum(confusion_vector == 1).item()
    FP = torch.sum(confusion_vector == float('inf')).item()
    TN = torch.sum(torch.isnan(confusion_vector)).item()
    FN = torch.sum(confusion_vector == 0).item()
    correct = (output == target).float().sum().item()

    # compute the result
    ACC = 100 * correct / output.shape[0]
    TNR = 100 * TN / (TN + FP) if (TN + FP) != 0 else 0
    FPR = 100 * FP / (TN + FP) if (TN + FP) != 0 else 0
    FNR = 100 * FN / (TP + FN) if (TP + FN) != 0 else 0
    TPR = 100 * TP / (TP + FN) if (TP + FN) != 0 else 0
    return ACC, TNR, FPR, FNR, TPR

if __name__ == "__main__":
    design_id = '8t1'

    testdata_path = '../data/ispd2018/' + design_id + '/all_data_Standard.npy'
    model_path = './DNN_ispd2019/DNN_ispd2019_Adam_0.1.pt'

    device = (torch.device("cuda") if torch.cuda.is_available() else ("cpu"))

    # Load test data
    test_data = np.load(testdata_path)
    inputs = torch.from_numpy(test_data[:, 0:23])
    labels = torch.from_numpy(test_data[:, 23:])

    # Load model
    loaded_model = DNN().to(device=device)
    loaded_model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        input = inputs.to(device=device).float()
        labels = labels.to(device=device).long()
        labels = torch.squeeze(labels)
        outputs = loaded_model(input)
        _, predicted = torch.max(outputs, dim=1)

    ACC, TNR, FPR, FNR, TPR = model_eval(predicted, labels)
    print('ACC:{}, TNR:{}, FPR:{}, FNR:{}, TPR:{}'.format(ACC, TNR, FPR, FNR, TPR))

    annotation = 'ACC: %.5f, TNR: %.5f, FPR: %.5f, FNR: %.5f, TPR: %.5f' % (ACC, TNR, FPR, FNR, TPR)

    vis_labels = np.load('../data/ispd2018/' + design_id + '/label.npy')
    vis_labels[vis_labels >= 1] = 1
    # ground_truth = Image.fromarray(np.uint8(vis_labels),'L')
    # plt.imshow(ground_truth)
    # plt.show()

    x = vis_labels.shape[0]
    y = vis_labels.shape[1]

    pre_labels = predicted.cpu().detach().numpy()
    pre_labels = pre_labels.reshape(x, y)

    vis_labels = np.flip(vis_labels.T, axis=0)
    pre_labels = np.flip(pre_labels.T, axis=0)

    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    fig = plt.figure(figsize=(width / 100., height / 100.), dpi=100)
    ax = fig.add_subplot(1, 2, 1)
    plt.title("ground_truth")
    ax.matshow(vis_labels, cmap=plt.cm.Reds)

    ax = fig.add_subplot(1, 2, 2)
    plt.title("Prediction")
    plt.text(0, y + 10, annotation, fontsize=10, color='black')
    ax.matshow(pre_labels, cmap=plt.cm.Reds)

    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()

    # Save the image
    fig.savefig('./' + design_id + '.png')

    plt.show()





