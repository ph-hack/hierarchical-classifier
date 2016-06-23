from unittest import TestLoader, TextTestRunner
from unittest.case import TestCase
from distances import dtw, p2p_dist, dtw_gradient
from clustering import SmartDBSCAN, FastDSM, load_digits, StandardScaler, LDA, distance_matrix, load_iris, PCA
import numpy as np
import scipy.spatial.distance as dist
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import copy as cp
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC


class NoneTransformation(object):

    def __init__(self):
        pass

    def transform(self, X):

        return X


class SpaceRegion(object):

    def __init__(self, data, label, maxes=None, mins=None):

        if maxes is None and mins is None:
            self.mins = data.min(0)
            self.maxes = data.max(0)
        else:
            self.mins = mins
            self.maxes = maxes

        self.data = data
        self.label = label

    def __str__(self):

        if self.is_empty:
            type = 'Empty '

        elif self.is_homogeneous:
            type = 'Homogeneous '

        else:
            type = 'Mixed '


        return type + \
            ', '.join(['{}: {}->{}'.format(d, m, n)
                          for d, m, n in
                          zip(range(len(self.mins)), self.mins, self.maxes)])

    def __repr__(self):

        if self.is_empty:
            type = 'Empty\n'

        elif self.is_homogeneous:
            type = 'Homogeneous\n'

        else:
            type = 'Mixed\n'


        return 'Space Region\n' + type + \
            '\n'.join(['dim {}: min = {}, max = {}, dist = {}'.format(d, m, n, k)
                          for d, m, n, k in
                          zip(range(len(self.mins)), self.mins, self.maxes, self.maxes - self.mins)]) \
               + '\n'

    def __eq__(self, other):
        # TODO: Finish the implementation of this method
        pass

    @property
    def is_homogeneous(self):

        return len(np.unique(self.label)) == 1 and not self.is_empty

    @property
    def is_empty(self):

        return len(self.data) == 0

    def divide(self):

        quadrants = []
        m,n = self.data.shape
        combs = np.zeros((np.power(2, n), n))

        for i, c in enumerate(combs):

            binary = bin(i)[2:]
            binary = list(binary)
            # print binary

            for j, b in enumerate(binary):

                c[len(c)-len(binary)+j] = int(b)
                # print 'j =', j, ' b =', b

        halfs = (self.maxes - self.mins)/2. + self.mins

        # print combs
        # print halfs

        for c in combs:

            x = np.array([True] * m)
            mins = []
            maxes = []

            for d in range(n):

                if len(x) == 0:

                    print self

                if c[d] == 0:
                    x = (self.data[:,d] <= halfs[d]).__and__(x)
                    mins.append(self.mins[d])
                    maxes.append(halfs[d])
                else:
                    x = (self.data[:,d] > halfs[d]).__and__(x)
                    mins.append(halfs[d])
                    maxes.append(self.maxes[d])

            quad = SpaceRegion(self.data[x,:], self.label[x], np.array(maxes), np.array(mins))

            quadrants.append(quad)

        return quadrants

    def belongs(self, point):

        dims_tests = [point[d] >= self.mins[d] and point[d] < self.maxes[d]
                      for d in range(len(self.maxes))]

        return all(dims_tests)

    def is_mergeable(self, other, common_dim=None):

        min_dim = len(self.mins) - 1

        if common_dim is None:

            common_dim = self.common_dim(other)

        return sum(common_dim) == min_dim

    def common_dim(self, other):

        return [1 if self.mins[d] == other.mins[d] and self.maxes[d] == other.maxes[d] else 0
                for d in range(len(self.mins))]

    def merge(self, other):

        common_dim = self.common_dim(other)

        if self.is_mergeable(other, common_dim):

            for i, c in enumerate(common_dim):

                if c == 0:

                    self.maxes[i] = max(self.maxes[i], other.maxes[i])
                    self.mins[i] = min(self.mins[i], other.mins[i])

                    if len(other.data) > 0:
                        try:
                            self.data = np.concatenate((self.data, other.data))
                            self.label = np.concatenate((self.label, other.label))

                        except TypeError:

                            print 'current data =', self.data.shape
                            print 'current label =', self.label.shape
                            print 'other data =', self.data.shape
                            print 'othe label =', self.label.shape
                            raise TypeError('haha')

            return True

        return False


class QuadTreeNode(object):

    def __init__(self, id=-1, obj=None):

        self.id = id
        self.obj = obj
        self.label = None
        self.children = []
        self.parent = None

        if obj.is_homogeneous:

            self.label = obj.label[0]

        elif obj.is_empty:

            self.label = -1

    def __str__(self):

        return 'Class:[{}]\nObj: {}{}{}\nchildren\n\t{}'.format(self.label, '{', self.obj, '}', self.children)

    def __repr__(self):

        return str(self)

    def get_brothers(self):

        if self.parent is None:

            return []

        candidates_brothers = self.parent.children
        brothers = []

        for b in candidates_brothers:

            if b.obj != self.obj:

                brothers.append(b)

        return brothers

    def add_child(self, child):

        self.children.append(child)
        child.parent = self

    @property
    def isleaf(self):

        return len(self.children) == 0

    @property
    def isroot(self):

        return self.parent is None

    def divide(self):

        if self.label is None:

            children = self.obj.divide()

            for c in children:

                child = QuadTreeNode(obj=c)
                self.add_child(child)

        return self

    def merge(self, other):

        worked = self.obj.merge(other.obj)

        if worked:

            self.label = max(self.label, other.label)

        return worked

    def belongs(self, point):

        return self.obj.belongs(point)


class HierarchicalClassifier(object):

    def __init__(self, base_metric=dist.euclidean, levels=1, repr_method='mean', dim_transf='none'):

        self.root = None
        self.metric = base_metric
        self.levels = levels
        self.repr_method = repr_method
        self.representants = None
        self.dim_transf = dim_transf

    # TODO: Implement and test the PCA or LDA transformation after the distance one
    def fit(self, X, y):

        if self.levels > 1:

            warnings.warn('Hierarchical Classifier:\n Only one level is supported yet!')

        X = self._transform(X, y)

        print X.shape

        # cluster = SmartDBSCAN(metric=dist.cosine).fit(self.representants, [], [], True)

        self.root = QuadTreeNode(obj=SpaceRegion(X, y))

        queue = [self.root]

        while len(queue) >= 1:

            current_node = queue.pop()

            current_node.divide()

            if not current_node.isleaf:

                queue.extend(current_node.children)

        self._merge_nodes()

        return self

    #TODO: Implement the predict method
    def predict(self, X, force_class=False):

        X = self._transform(X)

        preds = []
        for i, x in enumerate(X):

            queue = [self.root]
            preds.append(-1)

            while len(queue) > 0:

                current_node = queue.pop()

                if current_node.belongs(x):

                    if not current_node.isleaf:

                        queue.extend(current_node.children)
                    else:
                        preds[i] = current_node.label
                        break

            if preds[i] == -1 and force_class:

                preds[i] = x.argmin()

        return np.array(preds)

    def _merge_nodes(self):

        queue = cp.copy(self.root.children)

        while len(queue) > 0:

            current_node = queue.pop()

            if current_node.label is None:

                queue.extend(cp.copy(current_node.children))

            else: #if current_node.label >= 0:
                brothers = current_node.get_brothers()

                for b in brothers:

                    if ((current_node.label == -1 and b.label >= 0) or (current_node.label >= 0  and (current_node.label == b.label or b.label == -1))) and current_node.merge(b):

                        # print 'Node \n{}\ntook over\n{}\n'.format(current_node, b)
                        b.parent.children.remove(b)

                        try:
                            queue.remove(b)

                        except ValueError as e:
                            warnings.warn(e.message)

                        queue.append(current_node)
                        break

    def _pick_representants(self, X, y):

        dm = distance_matrix(X, self.metric)
        classes = np.unique(y)

        representants = []

        for c in classes:

            Xc = np.where(y == c)[0]
            Xnc = np.where(y != c)[0]

            if self.repr_method == 'mean':

                r = dm[Xc,Xc].sum(0).argmin()
                rep = X[Xc[r],:]

            elif self.repr_method == 'closest':

                min_d = np.inf
                rep = None

                for x in Xc:

                    d = dm[x,Xnc].mean()

                    if d < min_d:

                        min_d = d
                        rep = X[x,:]

            elif self.repr_method == 'farthest':

                max_d = 0.
                rep = None

                for x in Xc:

                    d = dm[x,Xnc].mean()

                    if d > max_d:

                        max_d = d
                        rep = X[x,:]

            representants.append(rep)

        representants = np.array(representants)

        # print representants
        return representants

    def _transform(self, X, y=None):

        if y is not None:

            c = len(np.unique(y))

            if self.dim_transf == 'PCA':

                self.dim_transf = PCA(n_components=c) .fit(X, y)

            elif self.dim_transf == 'LDA':

                self.dim_transf = LDA(n_components=c).fit(X, y)

            elif self.dim_transf == 'scaler':

                self.dim_transf = StandardScaler().fit(X, y)

            elif self.dim_transf == 'hybrid':

                self.dim_transf = Pipeline(steps=[
                    ('pca', PCA()),
                    ('lda', LDA(n_components=c))
                ]).fit(X, y)

            else:
                self.dim_transf = NoneTransformation()

        X = self.dim_transf.transform(X)

        if y is not None:

            if isinstance(self.metric, FastDSM):

                self.metric.fit(X, y, True)

            self.representants = self._pick_representants(X, y)

        new_data = np.zeros((len(X), len(self.representants)))

        for i, d in enumerate(X):

            for j, r in enumerate(self.representants):

                new_data[i,j] = self.metric(d, r)

        return new_data


def dec_to_bin(x):

    return int(bin(x)[2:])


class HierarchicalTests(TestCase):

    def est_00_classifier_augmentation(self):

        # TODO: Implement the unit tests for the classifier using data augmentation
        pass

    def est_04_space_transformation(self):

        # db = load_digits(2)
        db = load_iris()
        # pca = PCA(n_components=2)
        # data = pca.fit(db.data, db.target).transform(db.data)
        data = db.data
        labels = db.target

        print data.shape

        # scaler = StandardScaler().fit(data, labels)
        scaler = None
        dsm = FastDSM(beta=0.1, alpha=0.1, scaler=scaler).fit(data, labels, True)
        # dsm.w = [1., 0.01647817, 0.01642334, 0.01642327, 0.01642329, 0.01642327
        #     , 0.01642388, 1., 1., 0.01642398, 0.01642325, 0.01642329
        #     , 0.01642328, 0.01642326, 0.0164235, 1., 0.01657487, 0.01642338
        #     , 0.01642325, 0.01642325, 0.01642324, 0.01642326, 0.01642341, 1.
        #     , 0.01703409, 0.01642334, 0.01642327, 0.01642324, 0.01642324, 0.01642326
        #     , 0.01642334, 1., 1., 0.01642336, 0.01642326, 0.01642324
        #     , 0.01642324, 0.01642326, 0.01642331, 1., 1., 0.01642348
        #     , 0.01642325, 0.01642324, 0.01642324, 0.01642326, 0.01642331, 1., 1.
        #     , 0.01642441, 0.01642325, 0.01642326, 0.01642329, 0.01642325, 0.01642333
        #     , 0.01642404, 1., 0.01669356, 0.01642336, 0.01642327, 0.01642337
        #     , 0.01642326, 0.01642331, 0.01642341]
        # dsm.w = [0.22876873, 0.22877008, 0.22876814, 0.228769]
        # dsm.w = [0.18530247, 0.18530217, 0.18530425, 0.18530243]
        # dsm.w = [0.07976648, 0.07976682]

        distance = dsm

        model = HierarchicalClassifier(2, distance=distance)
        model.fit(data, labels)

        dm = distance_matrix(data, distance)
        classes = np.unique(labels)

        representants = []

        for c in classes:

            Xc = np.where(labels == c)[0]
            Xnc = np.where(labels != c)[0]

            max_d = 10000.
            rep = None

            for x in Xc:

                d = dm[x,Xnc].mean()

                if d < max_d:

                    max_d = d
                    rep = data[x,:]

            representants.append(rep)
            # representants.append(data[Xc[0],:])

        representants = np.array(representants)

        representants = model.get_samples(model.root.children)
        print representants

        new_data = np.zeros((len(data), len(representants)))

        for i, d in enumerate(data):

            for j, r in enumerate(representants):

                new_data[i,j] = model.distance(d, r)

        print new_data.shape

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(new_data[np.where(labels == 0)[0], 0], new_data[np.where(labels == 0)[0], 1], new_data[np.where(labels == 0)[0], 2], c='r', marker='o')
        ax.scatter(new_data[np.where(labels == 1)[0], 0], new_data[np.where(labels == 1)[0], 1], new_data[np.where(labels == 1)[0], 2], c='b', marker='o')
        ax.scatter(new_data[np.where(labels == 2)[0], 0], new_data[np.where(labels == 2)[0], 1], new_data[np.where(labels == 2)[0], 2], c='y', marker='o')
        #
        # ax.scatter(new_data[np.where(labels == 0)[0], 0], new_data[np.where(labels == 0)[0], 1], c='r', marker='o')
        # ax.scatter(new_data[np.where(labels == 1)[0], 0], new_data[np.where(labels == 1)[0], 1], c='b', marker='o')

        # plt.plot(range(len(np.where(labels == 0)[0])), data[np.where(labels == 0)[0], 0], 'ro')
        # plt.plot(range(len(np.where(labels == 1)[0])), data[np.where(labels == 1)[0], 0], 'bo')
        # plt.plot(representants, 'g*')
        plt.show()

        # _, axes = plt.subplots(1, 2)
        # axes[0].imshow(representants[0,:].reshape((8,8)), cmap='gray', interpolation='none')
        # axes[1].imshow(representants[1,:].reshape((8,8)), cmap='gray', interpolation='none')
        # plt.show()

    def est_05_space_region(self):

        db = load_iris()

        data = db.data[:,:3]
        label = db.target

        q = SpaceRegion(data, label)

        print q.is_homogeneous

        quads = q.divide()

        print dec_to_bin(33)

        print q

        for c in quads:

            print c

        print 'it belongs =', q.belongs([5, 3, 2])
        print 'it belongs =', q.belongs([2, 1, 7])

    def test_06_classifier2(self):

        # TODO: Test with the iris and diabetes datasets
        db = load_digits()
        # db = load_iris()
        # pca = PCA(n_components=2)
        # data = pca.fit(db.data, db.target).transform(db.data)

        n = len(db.images)
        X = db.images.reshape((n, -1))
        y = db.target

        # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4)#, stratify=y)

        X_tr = X[:n/2]
        X_te = X[n/2:]
        y_tr = y[:n/2]
        y_te = y[n/2:]

        model = HierarchicalClassifier(dim_transf='PCA', base_metric=dist.euclidean, repr_method='mean')
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te, True)

        print classification_report(y_te, y_pred)
        print 'accuracy =', accuracy_score(y_te, y_pred)

        # lda = LDA(n_components=10).fit(X_tr, y_tr)
        # X_tr = lda.transform(X_tr)
        # X_te = lda.transform(X_te)
        #
        # svm = SVC().fit(X_tr, y_tr)
        #
        # y_pred = svm.predict(X_te)
        #
        # print classification_report(y_te, y_pred)
        # print 'accuracy =', accuracy_score(y_te, y_pred)

    def est_07_merging(self):

        data = np.array([
            [0, 4],
            [1, 6],
            [2, 5],
            [2, 8],
            [3, 6],
            [4.7, 6],
            [3.5, 0],

            [3.7, 3.8],
            [4.9, 2.4],
            [6, 3.5],
            [7, 3.3],
            [5.3, 6],
            [6, 5.5],
            [7.6, 5.8],
            [6.3, 8],
            [7.3, 10],
            [10, 5.1],
        ])
        label = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])

        model = HierarchicalClassifier(dim_transf='hybrid')
        model.fit(data, label)

        plt.plot(data[:7,0],data[:7,1], 'ro')
        plt.plot(data[7:,0],data[7:,1], 'b*')
        plt.show()

        data = model.root.obj.data
        plt.plot(data[:7,0],data[:7,1], 'ro')
        # plt.plot(data[:7,0], 'ro')
        plt.plot(data[7:,0],data[7:,1], 'b*')
        # plt.plot(data[7:,0], 'b*')
        plt.show()

        test = np.array([
            [1, 1],
            [4, 1],
            [7, 7],
            [1, 10],
            [12, 6],
            [3, 14]
        ])

        # print model.predict(test)
        print model.predict(test, True)
        print np.array([0, 0, 1, 1, 1, 0])


if __name__ == '__main__':

    suite = TestLoader().loadTestsFromTestCase(HierarchicalTests)
    TextTestRunner(verbosity=2).run(suite)
