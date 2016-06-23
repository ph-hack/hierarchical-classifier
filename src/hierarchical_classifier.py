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
            homogeneous = 'Empty\n'

        elif self.is_homogeneous:
            homogeneous = 'Homogeneous\n'

        else:
            homogeneous = 'Mixed\n'


        return homogeneous + \
            '\n'.join(['dim {}: min = {}, max = {}'.format(d, m, n)
                          for d, m, n in
                          zip(range(len(self.mins)), self.mins, self.maxes)]) \
               + '\n'

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

        return '{}{}\nchildren\n\t{}{}'.format('{', self.obj, self.children, '}')

    def __repr__(self):

        return str(self)

    def get_brothers(self):

        if self.parent is None:

            return []

        candidates_brothers = self.parent.children
        brothers = []

        for b in candidates_brothers:

            if b.sample != self.obj:

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


class HierarchicalClassifier(object):

    def __init__(self, base_metric=dist.euclidean, levels=1, repr_method='mean'):

        self.root = None
        self.metric = base_metric
        self.levels = levels
        self.repr_method = repr_method

    def fit(self, X, y):

        if self.levels > 1:

            warnings.warn('Hierarchical Classifier:\n Only one level is supported yet!')

        X = self._transform(X, y)

        self.root = QuadTreeNode(obj=SpaceRegion(X, y))

        queue = [self.root]

        while len(queue) >= 1:

            current_node = queue.pop()

            current_node.divide()

            if not current_node.isleaf:

                queue.extend(current_node.children)

        return self

    #TODO: Implement the merge method to remove the empty nodes
    #TODO: Implement the predict method

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

    def _transform(self, X, y):

        representants = self._pick_representants(X, y)

        new_data = np.zeros((len(X), len(representants)))

        for i, d in enumerate(X):

            for j, r in enumerate(representants):

                new_data[i,j] = self.metric(d, r)

        # print new_data.shape
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

        db = load_digits(2)
        # db = load_iris()
        # pca = PCA(n_components=2)
        # data = pca.fit(db.data, db.target).transform(db.data)
        data = db.data
        labels = db.target

        model = HierarchicalClassifier()
        model.fit(data, labels)

        print model.root


if __name__ == '__main__':

    suite = TestLoader().loadTestsFromTestCase(HierarchicalTests)
    TextTestRunner(verbosity=2).run(suite)
