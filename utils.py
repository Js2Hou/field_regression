import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=True)
def kernel_value(A, B):
    value = 0
    p = 50
    H, W = A.shape
    w_patch = 4
    h_patch = 4
    w_step = 2
    h_step = 2
    for i in range(0, H - h_patch, h_step):
        for j in range(0, W - w_patch, w_step):
            cur_A = A[i:i + h_patch - 1, j:j + w_patch - 1]
            cur_B = B[i:i + h_patch - 1, j:j + w_patch - 1]
            value = value + (0.5 * (np.sum(cur_A * cur_B) / np.linalg.norm(cur_A) / np.linalg.norm(cur_B) + 1)) ** p
    return value


@nb.jit(nopython=True, parallel=True)
def get_kernel_vector(A, b):
    _, _, N = A.shape
    k_bA = np.zeros((N, 1))
    for i in range(N):
        k_bA[i] = kernel_value(A[:, :, i], b)
    return k_bA


@nb.jit(nopython=True, parallel=True)
def get_kernel_matrix(A):
    _, _, N = A.shape
    F = np.eye(N, N)
    for i in range(N):
        for j in range(N):
            F[i, j] = kernel_value(A[:, :, i], A[:, :, j])
    return F


@nb.jit(nopython=True, parallel=True)
def field_regression(A, B, pre_inverse):
    _, _, N = A.shape
    k_bA = np.zeros((N, 1))
    for i in range(N):
        k_bA[i] = kernel_value2(A[:, :, i], B)
    x = pre_inverse @ k_bA
    return x


@nb.jit(nopython=True, parallel=True)
def kernel_value2(A, B):
    value = 0
    p = 5
    H, W = A.shape
    w_patch = 8
    h_patch = 8
    w_step = 8
    h_step = 8
    for i in range(0, H - h_patch, h_step):
        for j in range(0, W - w_patch, w_step):
            cur_A = A[i:i + h_patch - 1, j:j + w_patch - 1]
            cur_B = B[i:i + h_patch - 1, j:j + w_patch - 1]
            value = value + (0.5 * (np.sum(cur_A * cur_B) / np.linalg.norm(cur_A) / np.linalg.norm(cur_B) + 1)) ** p
    return value
