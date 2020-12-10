import numpy as np
from scipy.io import loadmat

class_count = 100
train_count = 6
test_count = 100
name = ['book', 'glasses', 'hand', 'illumin', 'sarf', 'test']
count = [3, 3, 7, 3, 3, 6]

data_path = r'./Data_indoor.mat'
dataset = loadmat(data_path)
DAT = [dataset['Test_book'], dataset['Test_glasses'], dataset['Test_hand'], dataset['Test_illumin'],
       dataset['Test_sarf'], dataset['Train_DAT']]
train_dat = dataset['Train_DAT']

for i in range(5):
    right_num = 0
    for j in range(count[i]):
        for k in range(test_count):
            test_sample = DAT[i]
            min_dist = 9e9
            min_index = 0
            for l in range(class_count):
                for q in range(train_count):
                    dist = np.linalg.norm(train_dat[:, :, q, l] - test_sample[:, :, j, k], ord=1)
                    if dist < min_dist:
                        min_index = l
                        min_dist = dist
            if min_index == k:
                right_num += 1
    acc = right_num / test_count / count[i]
    print(f'acc for {name[i]} is {acc}\n')
