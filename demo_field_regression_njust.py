import numpy as np

from data_prepare import load_data
from utils import get_kernel_matrix, get_kernel_vector

test_scenarios_names = ['book', 'glasses', 'hand', 'illumin', 'sarf', 'test']

n_classes, xs_train, ys_train, xs_test_scenarios, ys_test_scenarios = load_data()

n_train = xs_train.shape[0]

lambdas = [0.5, 1, 10]
# kernel_matrix_path = r'./intermediate_matrices/kernel_matrix.mat'
# if os.path.exists(kernel_matrix_path):
#     mat = loadmat(kernel_matrix_path)
#     f = mat['kernel_matrix']
# else:
#     f = get_kernel_matrix(xs_train)
#     savemat(kernel_matrix_path, {'kernel_matrix': f})

f = get_kernel_matrix(xs_train)

for lamb in lambdas:  # 超参数选择
    print(f'lambda = {lamb}\n')
    # intermediate_matrix_path = f'./intermediate_matrices/lamb_{lamb}_paras.mat'
    # if os.path.exists(intermediate_matrix_path):
    #     mat = loadmat(intermediate_matrix_path)
    #     pre_inverse = mat['pre_inverse']
    # else:
    #     pre_inverse = np.linalg.inv(f + lamb * np.eye(n_train, n_train))
    #     savemat(intermediate_matrix_path, {'pre_inverse': pre_inverse})
    pre_inverse = np.linalg.inv(f + lamb * np.eye(n_train, n_train))
    print(f'inverse finished...\n')

    i = 0
    for xs_test, ys_test in zip(xs_test_scenarios, ys_test_scenarios):
        right_num = 0
        for x_test, y_test in zip(xs_test, ys_test):
            v = get_kernel_vector(xs_train, x_test)
            x = pre_inverse @ v
            err = np.zeros((n_classes, 1))
            for l in range(n_classes):
                x_hat = x[ys_train == l]
                f_hat = f[ys_train == l, :][:, ys_train == l]
                v_hat = v[ys_train == l]
                err[l] = x_hat.T @ f_hat @ x_hat - 2 * v_hat.T @ x_hat  # @为矩阵乘法，点乘用*
            y_pred = np.argmin(err)
            if y_test == y_pred:
                right_num += 1
        acc = right_num / xs_test.shape[0]
        print(f'acc for {test_scenarios_names[i]} is {acc}\n')
        i += 1
