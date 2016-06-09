from unittest import TestLoader, TextTestRunner
from unittest.case import TestCase
from distances import dtw, p2p_dist, dtw_gradient
from clustering import SmartDBSCAN, FastDSM, load_digits, StandardScaler, LDA, distance_matrix, load_iris, PCA
import numpy as np
import scipy.spatial.distance as dist
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HierarchicalClassifier:

    def __init__(self, level=1, distance=dist.euclidean, augmentations=None, log_file='classifier.log'):

        self.level = level
        self.root = HierarchicalNode('root')
        self.distance = distance
        self.augmentations = augmentations

        logging.basicConfig(format='%(asctime)s %(message)s', filename=log_file, level=logging.DEBUG, filemode='w')

    def fit(self, X, y):

        # makes all samples nodes
        leafs = self._make_nodes(X, y)
        last_level = leafs

        if self.level >= 2:

            last_level = self.build_level(last_level, y)

        for l in range(2, self.level):

            eps = []
            min_samples = []

            data = self.get_samples(last_level)

            model = SmartDBSCAN(self.distance)
            model.fit(data, eps, min_samples, True)

            last_level = self.build_level(last_level, model.labels_)

        for n in last_level:

            self.root.add_child(n)

        self.root.compute_stats()

    def get_samples(self, level):

        samples = []

        for node in level:

            samples.append(node.sample)

        return np.array(samples)

    def build_level(self, last_level, y):

        this_level = []
        classes = np.unique(y)

        for c in classes:

            Xc = np.where(y == c)[0]

            node_c = HierarchicalNode(label=c, distance=self.distance)

            for n in Xc:

                node_c.add_child(last_level[n])

            node_c.compute_stats()

            this_level.append(node_c)

        return this_level

    def predict(self, X):
        pass

    def decision_function(self, X):

        decision = []
        i = 1

        logging.info('Applying classifier\n')

        for i,x in enumerate(X):

            logging.info('on sample {}----------------------------------'.format(i))

            node_queue = [self.root]

            candidates = {}
            closest_sample = None
            min_error = 100

            while(len(node_queue) > 0):

                current_node = node_queue.pop(0)

                if current_node.isleaf:

                    # error = current_node.sample - x
                    error = current_node - x

                    if current_node.label not in candidates:

                        candidates[current_node.label] = error

                    else:

                        candidates[current_node.label] = error \
                            if error < candidates[current_node.label] \
                            else candidates[current_node.label]

                    if error < min_error:

                        closest_sample = current_node.sample
                        min_error = error

                elif current_node.test(x):

                    node_queue.extend(current_node.children)

                else:

                    logging.warn('Did not passed {}'.format(current_node))

            if len(candidates) > 0:
                chosen = elect(candidates)
            else:
                chosen = [-1]

            decision.append(chosen[0])

            # face.compare_show(closestface, measure=True)
            logging.info('candidates:\n{}\n'.format(str(top_candidates(candidates, 10))))
            logging.info('face {}, closest {}\n'.format(str(x), str(closest_sample)))
            logging.info('complete {}%\n'.format(i*100./len(X)))
            i += 1

        return decision

    def _make_nodes(self, X, Y):

        nodes = []
        for i, x, y in zip(range(len(X)), X, Y):

            #applies data augmentation and creates the new nodes
            if self.augmentations is not None:

                # TODO: Make the augmentantion methods generic as well
                # faces = aug.augment_faces([face], self.augmentations)
                #
                # for f in faces:
                #
                #     nodes.append(HierarchicalNode(f))
                pass

            else:
                nodes.append(HierarchicalNode(i, x, y, self.distance))

        return nodes


class HierarchicalNode:

    def __init__(self, id=-1, sample=None, label=None, distance=dist.euclidean):

        self.id = id
        self.sample = sample
        self.label = label
        self.children = []
        self.parent = None
        self.stats = stats_template()
        self.distance = distance

    def __str__(self):

        return '{}({})'.format(self.id, self.label)

    def __repr__(self):

        root = str(self)
        parent = str(self.parent)
        children = str([str(c) for c in self.children])
        brothers = str([str(b) for b in self.get_brothers()])

        return ''.join(['root = ', root, '\nparent = ', parent, '\nbrothers = ', brothers, '\nchildren = ', children,
                        '\nstats:\n\tmax = ', str(self.stats['max']), '\n\tmean = ', str(self.stats['mean']),
                        '\n\tvar = ', str(self.stats['var'])])

    def __len__(self):

        return len(self.children)

    def __sub__(self, other):

        return self.distance_to(other)

    def get_brothers(self):

        if self.parent is None:

            return []

        candidates_brothers = self.parent.children
        brothers = []

        for b in candidates_brothers:

            if b.sample != self.sample:

                brothers.append(b)

        return brothers

    def add_child(self, child):

        self.children.append(child)
        child.parent = self

    def test(self, other):

        if self.stats['max'] == 0 and self.stats['mean'] == 0 and self.stats['var'] == 0:

            return True

        error = self - other

        return error <= self.stats['max']

    def compute_stats(self):

        samples = np.array([child.sample for child in self.children])

        distances = distance_matrix(samples, self.distance)
        s = distances.sum(0).argmin()

        # distances = np.array([self.distance(s, samples.mean(0)) for s in samples])
        # s = distances.argmin()

        self.sample = samples[s,:]

        maxs = [child.stats['max'] for child in self.children]
        maxs.append(distances.max())

        self.stats['max'] = np.max(maxs)
        self.stats['mean'] = distances.mean()
        self.stats['var'] = np.mean(np.abs(distances - self.stats['mean']))

    def get_children_max(self):

        return [child.stats['max'] for child in self.children]

    @property
    def isleaf(self):

        return len(self.children) == 0

    @property
    def isroot(self):

        return self.parent is None

    def distance_to(self, other):

        if not isinstance(other, HierarchicalNode):

            other = HierarchicalNode(sample=other)

        return self.distance(self.sample, other.sample)


def stats_template():

    return {

        'max': 0,
        'mean': 0,
        'var': 0
    }

def elect(candidates):

    keys = candidates.keys()
    values = np.array(candidates.values())

    v = values.argmin()

    return (keys[v], values[v])

def top_candidates(candidates, k=1, order=None):

    keys = np.array(candidates.keys())
    values = np.array(candidates.values())

    v = values.argsort().tolist()

    if order == 'desc':

        v.reverse()

    d = {}

    for i in v[:k]:

        d[keys[i]] = values[i]

    return d


class HierarchicalTests(TestCase):

    def est_01_node(self):

        n1 = HierarchicalNode(0, [1,2,3,4], 'a')
        n2 = HierarchicalNode(1, [5,6,7], 'b')
        n3 = HierarchicalNode(2, [8,9,10], 'c')

        n1.add_child(n2)
        n1.add_child(n3)

        reprn1 = "root = 0(a)\nparent = None\nbrothers = []\nchildren = ['1(b)', '2(c)']\nstats:\n\tmax = 0\n\tmean = 0\n\tvar = 0"
        reprn2 = "root = 1(b)\nparent = 0(a)\nbrothers = ['2(c)']\nchildren = []\nstats:\n\tmax = 0\n\tmean = 0\n\tvar = 0"

        # print 'n1:\n', repr(n1), '\n'
        # print 'n2:\n', repr(n2), '\n'

        self.assertEqual(repr(n1), reprn1)
        self.assertEqual(repr(n2), reprn2)

        self.assertTrue(n2.isleaf)
        self.assertFalse(n1.isleaf)
        self.assertTrue(n1.isroot)

    def est_02_classifier(self):

        db = load_digits(3)
        pca = LDA(n_components=2)
        data = pca.fit(db.data[:300,:], db.target[:300]).transform(db.data[:300,:])
        # data = db.data[:300,:]
        labels = db.target[:300]

        model = HierarchicalClassifier(2)
        model.fit(data, labels)

        print repr(model.root)

        print float(np.sum(np.array(model.decision_function(pca.transform(db.data[300:400,:]))) == labels))/len(db.target[300:400])

    def est_00_classifier_augmentation(self):

        # TODO: Implement the unit tests for the classifier using data augmentation
        pass

    def test_04_space_transformation(self):

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

    def est_05_(self):

        x = np.array([5, 6, 7])
        y = np.array([1, 2, 3])
        z = np.array([4, 8, 10])

        w = np.array([2., 1.3, 0.2])

        print 'x,y', dist.euclidean(x,y)/6.92
        print 'x,z', dist.euclidean(x,z)/6.92

        print 'xw,yw', dist.euclidean(x*w,y*w)/9.57
        print 'xw,zw', dist.euclidean(x*w,z*w)/9.57


if __name__ == '__main__':

    suite = TestLoader().loadTestsFromTestCase(HierarchicalTests)
    TextTestRunner(verbosity=2).run(suite)
