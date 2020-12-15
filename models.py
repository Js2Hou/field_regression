import cmath
import math

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ComplexPCA(object):
    """Complex principle component analysis

    Input
    --------
    X : array-like of shape (n_samples, n_features)
        X is a complex-valued matrix

    Parameter
    --------
    n_components : int or None
        Numbers of components to keep.
        if n_components is not set, then all components are kept:
            n_components == min(n_samples, n_features)

    threshold : float or None
        Lower bound constraint on the sum of the explained variance ratios of components. If n_components is specified,
        this parameter is ignored.

    Attributes
    --------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum variance in the data. The components
        are sorted by ``explained_variance_``.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` and ``total`` are not set then all components are stored and the sum of the ratios is equal
        to 1.0.

    n_components_ : int
        The estimated number of components. If n_components is not specified, program will auto compute n_components_
        by the given explained_variance_ratio_sum. Otherwise, it equals the parameter n_components.

    n_features_ : int
        Number of features in the training data.

    n_samples_ : int
        Number of samples in the training data.
    """

    def __init__(self, n_components=None, threshold=None):
        self.n_components = n_components
        self.threshold = threshold
        self.mmscale = MinMaxScaler()

    def data_process(self, X, y=None):
        # return self.mmscale.transform(X)
        return X

    def inverse_data_process(self, X, y=None):
        # return self.mmscale.inverse_transform(X)
        return X

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        --------
        X : array_like, shape (n_samples, n_features)
            Training data, where n_samples if the number of samples and n_features if the number of features

        y : None
            Ignored variable

        Returns
        --------
        self : object
            Returns the instance itself.
        """
        X = X.copy()
        self.mmscale.fit(X)
        self._fit(X)
        return self

    def _fit(self, X):
        """Fit the model by computing eigenvalue decomposition on X * X.H"""
        Z = self.data_process(X)

        n_samples, n_features = Z.shape

        # Handle n_components==None
        n_components = min(n_samples, n_features) if self.n_components is None else self.n_components

        # Use the eigenvalue decomposition method to obtain the transformation matrix B_H from x to y
        Z_H = np.conj(Z).T
        # K = Z_H @ Z
        # w, v = np.linalg.eigh(K)
        # w1 = np.flip(w).real
        # v1 = np.fliplr(v)
        # B = v1[:, :n_components]
        # B_H = np.conj(B).T

        K = Z @ Z_H
        w, v = np.linalg.eig(K)  # The eigenvalues w are not ordered
        order = np.argsort(w)
        w = np.sort(w)
        v = v[:, order]
        w1 = np.flip(w).real
        v1 = np.fliplr(v)

        U_m = v1[:, :n_components]
        Lambda_sqrt_minus_2 = np.diag(np.float_power(w1[:n_components], -0.5))
        B = Z_H @ U_m @ Lambda_sqrt_minus_2
        B_H = np.conj(B).T

        # Get variance explained by eigenvalues
        w1 = np.sqrt(w1)
        try:
            total_variance = w1.sum()
            explained_variance_ratio_ = w1 / total_variance
        except ZeroDivisionError:
            print('all eigenvalue of covariance of X are 0.')
            return ZeroDivisionError

        # Calculate the cumulative contribution rate of variance
        if self.n_components == min(n_samples, n_features) and self.threshold is not None:
            explained_variance_ratio_sum_ = 0
            for t, e in enumerate(explained_variance_ratio_):
                explained_variance_ratio_sum_ += e
                if explained_variance_ratio_sum_ >= self.threshold:
                    n_components = t + 1
                    break
        else:
            explained_variance_ratio_sum_ = sum(explained_variance_ratio_[:n_components])

        # Add instance attributes
        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = B_H[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.explained_variance_ratio_sum = explained_variance_ratio_sum_
        self.B_H = B_H[:n_components, :]
        self.B = B[:, :n_components]
        self.w = w

        return B_H, B

    def transform(self, Z):
        Z = Z.copy()
        Z = self.data_process(Z)
        return (self.B_H @ Z.T).T

    def inverse_transform(self, Y):
        Y = Y.copy()
        X = (self.B @ Y.T).T
        return self.inverse_data_process(X)


class EPCA(ComplexPCA):
    """Euler principal component analysis (ePCA).

    Parameter
    ----------
    alpha : flaot or None
        Parameter of euler transform.

    """

    def __init__(self, alpha=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.mmscale = MinMaxScaler()

    def data_process(self, X, y=None):
        # X = self.mmscale.transform(X)
        return self.euler(X)

    def inverse_data_process(self, X, y=None):
        X = self.inverse_euler(X)
        # return self.mmscale.inverse_transform(X)
        return X

    def euler(self, x):
        z = (cmath.e ** (1j * self.alpha * math.pi * x)) / math.sqrt(2)
        return z

    def inverse_euler(self, z):
        x = (np.angle(z) / (self.alpha * cmath.pi)).real
        return x  # return (-1j * np.log(np.sqrt(2) * z)).real


class KNN(object):
    """k-nearest neighbor

    """

    def __init__(self, train_x, test_x, train_labels, test_labels, k=5):
        self.train_x = train_x
        self.test_x = test_x
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.pred_labels = None
        self.accuracy = None
        self.k = k

    def fit(self):
        train_x = self.train_x.copy()
        test_x = self.test_x.copy()
        train_labels = self.train_labels
        test_labels = self.test_labels
        n = test_x.shape[0]
        right_num = 0
        pred_labels = []

        for id, y in enumerate(test_x):
            distance = np.array([np.linalg.norm(i - y) for i in train_x])
            distance_map_label = np.vstack((distance, train_labels)).T
            distance_map_label = distance_map_label[np.argsort(distance_map_label[:, 0])]
            distance_map_label = distance_map_label.astype('int64')
            label_pred = np.argmax(np.bincount(distance_map_label[:self.k, 1]))
            pred_labels.append(label_pred)
            if label_pred == test_labels[id]:
                right_num += 1
        accuracy = right_num / n
        pred_labels = np.array(pred_labels)
        self.pred_labels = pred_labels
        self.accuracy = accuracy
        return self
