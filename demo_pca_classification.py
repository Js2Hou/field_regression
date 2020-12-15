from sklearn.decomposition import KernelPCA

from data_prepare import load_data
from models import KNN

test_scenarios_names = ['book', 'glasses', 'hand', 'illumin', 'sarf', 'test']

_, xs_train, ys_train, xs_test_scenarios, ys_test_scenarios = load_data()

n_components = 10
k = 5

# we can change the model via comment one of the next two lines
pca = KernelPCA(n_components=n_components, kernel='rbf')
# pca = ComplexPCA(n_components=n_components)


n, h, w = xs_train.shape
xs_train = xs_train.reshape(n, h * w)
pca.fit(xs_train)
xs_train = pca.transform(xs_train)

i = 0
for xs_test, ys_test in zip(xs_test_scenarios, ys_test_scenarios):
    n, h, w = xs_test.shape
    xs_test = xs_test.reshape(n, h * w)
    xs_test = pca.transform(xs_test)
    knn = KNN(xs_train, xs_test, ys_train, ys_test, k)
    knn.fit()
    print(f'acc for {test_scenarios_names[i]} is {knn.accuracy}')
    i += 1
