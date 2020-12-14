import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle


def load_data(data_path=r'./NUST/Data_indoor.mat'):
    """
    Returns
        x_train : shape (n, h, w)
    """
    dataset = loadmat(data_path)
    DAT = [dataset['Test_book'], dataset['Test_glasses'], dataset['Test_hand'], dataset['Test_illumin'],
           dataset['Test_sarf'], dataset['Train_DAT']]

    x_test_scenarios = []
    y_test_scenarios = []
    x_train, y_train = 0, 0  # 没啥用，完全是因为没有这一行下面会警告，看着不舒服

    ratio = 0.1
    n_classes = int(DAT[0].shape[-1] * ratio)

    for i in range(len(DAT)):
        DAT[i] = DAT[i][:, :, :, :n_classes]
        h, w, n_imgs_one, _ = DAT[i].shape
        xs = DAT[i].reshape((h, w, n_imgs_one * n_classes), order='F')
        xs = xs.transpose(2, 1, 0)
        ys = np.hstack([[i for j in range(n_imgs_one)] for i in range(n_classes)])
        xs, ys = shuffle(xs, ys)
        if i == len(DAT) - 1:
            x_train, y_train = xs, ys
            break
        x_test_scenarios.append(xs)
        y_test_scenarios.append(ys)

    return n_classes, x_train, y_train, x_test_scenarios, y_test_scenarios
