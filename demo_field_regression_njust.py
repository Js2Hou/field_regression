import os

import numpy as np
from scipy.io import loadmat, savemat

from utils import get_kernel_matrix, get_kernel_vector

class_count = 100  # 训练集中人数
train_count = 6
test_count = 100  # 测试集中人数
name = ['book', 'glasses', 'hand', 'illumin', 'sarf', 'test']
count = [3, 3, 7, 3, 3, 6]  # len(count) == 6，表示5份测试集+1份训练集；里面数据表示每份数据集中有几张同一个人的图片

data_path = r'./NUST/Data_indoor.mat'
dataset = loadmat(data_path)
DAT = [dataset['Test_book'], dataset['Test_glasses'], dataset['Test_hand'], dataset['Test_illumin'],
       dataset['Test_sarf'], dataset['Train_DAT']]

train_dat = dataset['Train_DAT']
train_dat = train_dat.reshape((32, 32, 600), order='F')
train_dat = train_dat.transpose(2, 1, 0)
train_label = np.hstack([[i for j in range(6)] for i in range(100)])
# train_dat, train_label = shuffle(train_dat, train_label)
train_dat = train_dat.transpose(2, 1, 0)
n = 600

lambdas = [0.5, 1, 10]
kernel_matrix_path = r'./intermediate_matrices/kernel_matrix.mat'
if os.path.exists(kernel_matrix_path):
    mat = loadmat(kernel_matrix_path)
    f = mat['kernel_matrix']
else:
    f = get_kernel_matrix(train_dat)
    savemat(kernel_matrix_path, {'kernel_matrix': f})

for lamb in lambdas:  # 超参数选择
    print(f'lambda = {lamb}\n')
    intermediate_matrix_path = f'./intermediate_matrices/lamb_{lamb}_paras.mat'
    if os.path.exists(intermediate_matrix_path):
        mat = loadmat(intermediate_matrix_path)
        pre_inverse = mat['pre_inverse']
    else:
        pre_inverse = np.linalg.inv(f + lamb * np.eye(n, n))
        savemat(intermediate_matrix_path, {'pre_inverse': pre_inverse})
    print(f'inverse finished...\n')
    for i in range(5):  # 不同测试场景
        right_num = 0
        for j in range(count[i]):  # 测试集中同一个人照片数目
            for k in range(test_count):  # 测试集中总人数
                test_sample = DAT[i]
                v = get_kernel_vector(train_dat, test_sample[:, :, j, k])
                x = pre_inverse @ v
                err = np.zeros((class_count, 1))
                for l in range(class_count):  # 训练集中总人数
                    x_hat = x[train_label == l]
                    # a[[1,2],[1,2]]在python中取出的是第一行第一列和第二行第二列元素组成的向量，而在matlab中取出的是一二行与一二列交叉处的矩阵
                    f_hat = f[train_label == l, :][:, train_label == l]
                    v_hat = v[train_label == l]
                    # print(f'x_hat : {x_hat.shape} f_hat : {f_hat.shape} v_hat : {v_hat.shape}')
                    err[l] = x_hat.T @ f_hat @ x_hat - 2 * v_hat.T @ x_hat  # @为矩阵乘法，点乘用*
                id = np.argmin(err)
                if k == id:
                    right_num += 1
        acc = right_num / test_count / count[i]
        print(f'acc for {name[i]} is {acc}\n')
