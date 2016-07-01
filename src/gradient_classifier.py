"""
Here should go a brief description of this code and how to use it:

"""

# here goes the imports
from unittest import TestLoader, TextTestRunner
from unittest.case import TestCase
import math
from clustering import distance_matrix
from numpy import where, array, inf, zeros, unique, histogram, linspace, concatenate, argmax
from scipy.interpolate import interp1d
from dsm import FastDSM
from math import ceil
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt



# your code goes here

class GradientClassifier(object):

    def __init__(self, base_distance=dist.euclidean, beta=0.1, alpha=1., repr_method='mean', w=1.):

        self.metric = base_distance
        self.beta = beta
        self.alpha = alpha
        self.w = w
        self.repr_method = repr_method
        self.gradients = []
        self.distributions = []
        self.functions = []
        self.repr = []
        self.repr_x = []

    def fit(self, X, y):

        dm = distance_matrix(X, self.metric)
        reprs, indexes = self._pick_representants(X, y, dm)
        self.repr = reprs
        self.repr_x = indexes

        classes = unique(y)
        classes.sort()

        for i, r in zip(indexes, reprs):

            repr_grads = []
            repr_dists = []
            repr_funcs = []

            for c in classes:

                Xc = where(y == c)[0]
                grads = []

                for j in Xc:

                    if j != i:
                        # initial error
                        e0 = dm[i,j]
                        # errors after beta change
                        e1s = []

                        # for each variable(feature) v
                        for v in range(len(X[i,:])):

                            temp_x = X[j,:].copy()
                            temp_x[v] += self.beta

                            e1s.append(self.metric(X[i,:], temp_x))
                            # e1s.append(X[i,v] - X[j,v])

                        e1s = array(e1s)
                        grads.append((e1s - e0)/self.beta)
                        # grads.append(e1s)

                grads = array(grads).transpose()
                dists = []
                fs = []

                for g in grads:

                    h, x = histogram(g, len(g))
                    # x = [(d1 + d2)/2. for d1, d2 in zip(x[:-1], x[1:])]
                    h = concatenate((h, array([h[-1]])))

                    h = h.astype(float)/h.max()

                    dists.append(array([x, h]).transpose())

                    f = interp1d(x, h)
                    new_x = linspace(min(x), max(x), self.alpha*len(x), endpoint=True)
                    f = interp1d(new_x, f(new_x), 2, fill_value=0, bounds_error=False)

                    fs.append(f)

                dists = array(dists)
                repr_grads.append(grads)
                repr_dists.append(dists)
                repr_funcs.append(fs)

            self.gradients.append(repr_grads)
            self.distributions.append(repr_dists)
            self.functions.append(repr_funcs)

        return self

    def predict(self, X):

        preds = []

        votes = self.decision_function(X)

        for v in votes:

            preds.append(argmax(v))
            # print v

        return array(preds)

    def decision_function(self, X):

        Nc = len(self.repr)
        preds = []

        for x in X:

            votes = [[] for i in range(Nc)]

            for cg, class_grad in enumerate(self.gradients):

                for g, grad in enumerate(class_grad):

                    for v, variable in enumerate(grad):

                        e0 = self.metric(self.repr[cg], x)

                        temp_x = x.copy()
                        temp_x[v] += self.beta
                        e1 = self.metric(self.repr[cg], temp_x)

                        delta = (e1 - e0)/self.beta
                        # delta = self.repr[cg][v] - x[v]

                        prob = self.functions[cg][g][v](delta)

                        if cg == g:

                            prob *= self.w

                        votes[g].append(prob)

            preds.append([sum(v) for v in votes])

        return preds


    def _pick_representants(self, X, y, dm=None):

        if dm is None:
            dm = distance_matrix(X, self.metric)

        classes = unique(y)

        representants = []
        indexes = []

        for c in classes:

            Xc = where(y == c)[0]
            Xnc = where(y != c)[0]

            if self.repr_method == 'mean':

                r = dm[Xc,Xc].sum(0).argmin()
                rep = X[Xc[r],:]
                ind = Xc[r]

            elif self.repr_method == 'closest':

                min_d = inf
                rep = None

                for x in Xc:

                    d = dm[x,Xnc].mean()

                    if d < min_d:

                        min_d = d
                        rep = X[x,:]
                        ind = x

            elif self.repr_method == 'farthest':

                max_d = 0.
                rep = None

                for x in Xc:

                    d = dm[x,Xnc].mean()

                    if d > max_d:

                        max_d = d
                        rep = X[x,:]
                        ind = x

            representants.append(rep)
            indexes.append(ind)

        representants = array(representants)

        # print representants
        return representants, array(indexes)

class FactorClassifier(object):

    def __init__(self, alpha=1., beta=0.001, base_metric=dist.euclidean, repr_method='mean', w=1.):

        self.metric = base_metric
        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.repr_method = repr_method
        self.factors = []
        self.distributions = []
        self.functions = []
        self.repr = []
        self.repr_x = []

    def fit(self, X, y):

        dm = distance_matrix(X, self.metric)
        reprs, indexes = self._pick_representants(X, y, dm)
        self.repr = reprs
        self.repr_x = indexes

        classes = unique(y)
        classes.sort()

        for i, r in zip(indexes, reprs):

            repr_factors = []
            repr_dists = []
            repr_funcs = []

            for c in classes:

                Xc = where(y == c)[0]
                factors = []

                for j in Xc:

                    if j != i:

                        e1s = []

                        # for each variable(feature) v
                        for v in range(len(X[i,:])):

                            # e1s.append(r[v]/(X[j,v] + self.beta))
                            e1s.append(r[v] - X[j,v])

                        e1s = array(e1s)
                        factors.append(e1s)

                factors = array(factors).transpose()
                dists = []
                fs = []

                for g in factors:

                    h, x = histogram(g, int(ceil(self.beta*len(g))))
                    # x = [(d1 + d2)/2. for d1, d2 in zip(x[:-1], x[1:])]
                    h = concatenate((h, array([h[-1]])))

                    h = h.astype(float)/h.max()

                    dists.append(array([x, h]).transpose())

                    f = interp1d(x, h)
                    new_x = linspace(min(x), max(x), self.alpha*len(x), endpoint=True)

                    if len(new_x) == 2:

                        f = interp1d(new_x, f(new_x), 1, fill_value=0, bounds_error=False)
                    else:

                        f = interp1d(new_x, f(new_x), 2, fill_value=0, bounds_error=False)

                    fs.append(f)

                dists = array(dists)
                repr_factors.append(factors)
                repr_dists.append(dists)
                repr_funcs.append(fs)

            self.factors.append(repr_factors)
            self.distributions.append(repr_dists)
            self.functions.append(repr_funcs)

        return self

    def predict(self, X):

        preds = []

        votes = self.decision_function(X)

        for v in votes:

            preds.append(argmax(v))
            # print v

        return array(preds)

    def decision_function(self, X):

        Nc = len(self.repr)
        preds = []

        for x in X:

            votes = [[] for i in range(Nc)]

            for cg, class_grad in enumerate(self.factors):

                for g, grad in enumerate(class_grad):

                    for v, variable in enumerate(grad):

                        # factor = self.repr[cg][v]/(x[v] + self.beta)
                        factor = self.repr[cg][v] - x[v]

                        # print 'repr =', self.repr[cg]
                        # print 'x =', x
                        # print 'factor =', factor
                        # print self.factors[cg][g][v]

                        prob = self.functions[cg][g][v](factor)

                        if cg == g:

                            prob *= self.w

                        # plt.plot(self.distributions[cg][g][v][:,0],
                        #          self.distributions[cg][g][v][:,1], 'b-')
                        # plt.plot(factor, self.functions[cg][g][v](factor), 'ro')
                        # plt.title('{},{}; v={} = {}'.format(cg, g, v, prob))
                        # plt.show()

                        votes[g].append(prob)

            preds.append([sum(v) for v in votes])

        return preds


    def _pick_representants(self, X, y, dm=None):

        if dm is None:
            dm = distance_matrix(X, self.metric)

        classes = unique(y)

        representants = []
        indexes = []

        for c in classes:

            Xc = where(y == c)[0]
            Xnc = where(y != c)[0]

            if self.repr_method == 'mean':

                r = dm[Xc,Xc].sum(0).argmin()
                rep = X[Xc[r],:]
                ind = Xc[r]

            elif self.repr_method == 'closest':

                min_d = inf
                rep = None

                for x in Xc:

                    d = dm[x,Xnc].mean()

                    if d < min_d:

                        min_d = d
                        rep = X[x,:]
                        ind = x

            elif self.repr_method == 'farthest':

                max_d = 0.
                rep = None

                for x in Xc:

                    d = dm[x,Xnc].mean()

                    if d > max_d:

                        max_d = d
                        rep = X[x,:]
                        ind = x

            representants.append(rep)
            indexes.append(ind)

        representants = array(representants)

        # print representants
        return representants, array(indexes)


def generate_2d_map(boundaries, shape, model):

    dx = (boundaries['xmax'] - boundaries['xmin'])/shape[0]
    dy = (boundaries['ymax'] - boundaries['ymin'])/shape[1]

    img = zeros(shape)

    for j in range(shape[0]):
        for i in range(shape[1]):

            x = boundaries['xmin'] + j * dx
            y = boundaries['ymin'] + i * dy
            img[i,j] = model.predict(array([x,y]).reshape((1,2)))[0]
            # print i,j

    return img

# Unit Tests 
class GradientClassifierTests(TestCase):
    def assertEqual(self, first, second, msg=None):
        if first != second:
            e_msg = ''.join(
                ['\nExpected: ', str(second), ' Found: ', str(first)])
            print e_msg

        TestCase.assertEqual(self, first, second, msg)

    def est_1_gradients(self):

        import matplotlib.pyplot as plt
        from sklearn.datasets import make_blobs
        from distances import p2p_dist

        # data = array([
        #     [0, 4],
        #     [1, 6],
        #     [2, 5],
        #     [2, 8],
        #     [3, 6],
        #     [4.7, 6],
        #     [3.5, 0],
        #
        #     [3.7, 3.8],
        #     [4.9, 2.4],
        #     [6, 3.5],
        #     [7, 3.3],
        #     [5.3, 6],
        #     [6, 5.5],
        #     [7.6, 5.8],
        #     [6.3, 8],
        #     [7.3, 10],
        #     [10, 5.1],
        #     [9., 8.],
        #     [5., 9.],
        #     [8., 9.],
        #     [8., 8.],
        #     [7., 8.],
        #     [2., 9.],
        # ])
        # label = array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

        data, label = make_blobs(40, center_box=(2, 3), centers=1, random_state=5)
        data2, label2 = make_blobs(40, center_box=(6, 2), centers=1, random_state=5)

        data = concatenate((data, data2))
        label = concatenate((label, label2+1))

        # plt.plot(data[label==0,0], data[label==0,1], 'bo')
        # plt.plot(data[label==1,0], data[label==1,1], 'ro')
        # plt.show()
        # return

        # db = load_iris()
        # data = db.data[:,:2]
        # label = db.label

        model = GradientClassifier(base_distance=p2p_dist, beta=0.01, alpha=2).fit(data, label)

        p = array([6.3,8.])

        print model.predict(p.reshape((1,2)))
        print model.decision_function(p.reshape((1,2)))

        boundaries = dict(
            xmin=-1.,
            xmax=11.,
            ymin=-1.,
            ymax=11.
        )

        img = generate_2d_map(boundaries, (60,60), model)
        # plt.imshow(img, interpolation='none')
        # plt.plot(data[:7,0]*5, data[:7,1]*5, 'co')
        # plt.plot(data[7:,0]*5, data[7:,1]*5, 'mo')
        # plt.show()

        for cg, class_grad in enumerate(model.gradients):

            for g, grad in enumerate(class_grad):

                for v, variable in enumerate(grad):

                    _, axes = plt.subplots(2,2)

                    axes[0,0].plot(variable, 'bo-')
                    axes[0,0].set_title('cg {}, g {}, v {} data'.format(cg, g, v))

                    axes[1,0].hist(variable)
                    axes[1,0].set_title('cg {}, g {}, v {} distribution'.format(cg, g, v))

                    axes[1,1].plot(model.distributions[cg][g][v, :, 0], model.distributions[cg][g][v, :, 1], 'bo--')
                    axes[1,1].set_title('cg {}, g {}, v {} distribution'.format(cg, g, v))

                    f = model.functions[cg][g][v]
                    xnew = linspace(model.distributions[cg][g][v,:,0].min(),
                                    model.distributions[cg][g][v,:,0].max(), num=50, endpoint=True)
                    axes[1,1].plot(xnew, f(xnew), 'r-')

                    e0 = model.metric(model.repr[cg], p)
                    temp_p = p.copy()
                    temp_p[v] += model.beta
                    e1 = model.metric(model.repr[cg], temp_p)

                    x = (e1 - e0)/model.beta
                    # print p, temp_p
                    # print 'cg=', cg, ' g=', g
                    # print 'e0=', e0, ' e1=', e1, 'x=', x, 'f(x)=', f(x)

                    axes[1,1].plot(x, f(x), 'y^')
                    axes[1,1].plot([x,x], [0,max(f(xnew))], 'y--')

                    axes[0,1].imshow(img, interpolation='none')
                    data = (data + 1) * 5
                    p = (p + 1)*5
                    axes[0,1].plot(data[label==0,0], data[label==0,1], 'bo')
                    axes[0,1].plot(data[label==1,0], data[label==1,1], 'ro')
                    axes[0,1].plot(p[0], p[1], 'go')
                    model.repr[0] = (model.repr[0] + 1)*5
                    model.repr[1] = (model.repr[1] + 1)*5
                    axes[0,1].plot(model.repr[0][0], model.repr[0][1], 'c^')
                    axes[0,1].plot(model.repr[1][0], model.repr[1][1], 'm^')

                    plt.show()

    def est_2_interpolation(self):

        import matplotlib.pyplot as plt
        from scipy import interpolate
        import numpy as np
        x = np.linspace(0, 10, num=11, endpoint=True)
        y = np.cos(-x**2/9.0)
        f = interpolate.interp1d(x, y, 2)

        xnew = np.linspace(0, 10, num=41, endpoint=True)
        ynew = f(xnew)   # use interpolation function returned by `interp1d`
        plt.plot(x, y, 'o', xnew, ynew, '-')
        plt.show()

    def est_4_dumb_data(self):

        import matplotlib.pyplot as plt
        from sklearn.datasets import make_blobs
        from sklearn.datasets import load_iris, load_digits
        from sklearn.cross_validation import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.preprocessing import StandardScaler
        from clustering import PCA, LDA
        from sklearn.svm import SVC
        from hierarchical_classifier import HierarchicalClassifier, DynamicClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from distances import p2p_dist, dtw
        import pandas as pd
        import m_learn.mlproblem as mlp

        data, label = make_blobs(40, center_box=(0, 0, 0), n_features=3, centers=1, cluster_std=1, random_state=5)
        data2, label2 = make_blobs(40, center_box=(5, 5, 5), n_features=3, centers=1, cluster_std=1, random_state=5)
        data3, label3 = make_blobs(40, center_box=(10, 10, 10), n_features=3, centers=1, cluster_std=1, random_state=5)
        data4, label4 = make_blobs(40, center_box=(15, 15, 15), n_features=3, centers=1, cluster_std=1, random_state=5)

        data = concatenate((data, data2, data3, data4))
        label = concatenate((label, label2+1, label3+2, label4+3))

        print data.shape

        # plt.plot(data[:,2], data[:,1], 'bo')
        # plt.show()

        X_tr, X_te, y_tr, y_te = train_test_split(data, label, test_size=0.3, random_state=5)

        # scaler = StandardScaler().fit(X_tr)
        # X_tr = scaler.transform(X_tr)
        # X_te = scaler.transform(X_te)
        #
        # print 'computing the LDA'
        # lda = LDA(n_components=2).fit(X_tr, y_tr)
        # X_tr = lda.transform(X_tr)
        # X_te = lda.transform(X_te)
        # print 'done', X_tr.shape

        # my_d = FastDSM(base_metric=dist.euclidean)#.fit(X_tr, y_tr, verbose=True)

        distances = [dist.euclidean, dist.euclidean, p2p_dist, dist.canberra,
                     dist.chebyshev, dist.dice, dist.hamming, dist.matching,
                     dist.kulsinski, dist.russellrao, dist.rogerstanimoto,
                     dist.braycurtis, dist.sokalmichener, dist.yule]

        # model = GradientClassifier(base_distance=dist.euclidean,
        #                            beta=1,
        #                            alpha=2.,
        #                            repr_method='mean',
        #                            w=2.)
        model = FactorClassifier(base_metric=dist.euclidean,
                                 repr_method='mean',
                                 alpha=2,
                                 w=2.,
                                 beta=0.2)
        # model = DynamicClassifier(dim_transf='none',
        #                           base_metric=dist.euclidean,
        #                           repr_method='mean')
        # model = HierarchicalClassifier(dim_transf='none',
        #                                base_metric=dist.euclidean,
        #                                repr_method='mean')
        model.fit(X_tr, y_tr)

        # new_data = model.root.obj.data
        #
        # print new_data.shape
        #
        # plt.plot(new_data[:,0], new_data[:,1], 'bo')
        # plt.show()
        # plt.plot(new_data[:,1], new_data[:,2], 'bo')
        # plt.show()

        print y_te[1]
        y_pred = model.predict(X_te)#, force_class=True)


        print y_pred == y_te

        print classification_report(y_te, y_pred)
        print 'accuracy =', accuracy_score(y_te, y_pred)

        model = SVC().fit(X_tr, y_tr)
        # model = KNeighborsClassifier(1, metric=p2p_dist).fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        print classification_report(y_te, y_pred)
        print 'accuracy =', accuracy_score(y_te, y_pred)

    def test_3_classification(self):

        from sklearn.datasets import load_iris, load_digits
        from sklearn.cross_validation import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.preprocessing import StandardScaler
        from clustering import PCA, LDA
        from sklearn.svm import SVC
        from hierarchical_classifier import HierarchicalClassifier, DynamicClassifier
        import matplotlib.pyplot as plt
        from sklearn.neighbors import KNeighborsClassifier
        from distances import p2p_dist, dtw
        import pandas as pd
        import m_learn.mlproblem as mlp

        # data = pd.read_csv('/home/phack/Documents/Mestrado/Hierarchical_Classifier/dermathology.csv').values
        # data = pd.read_csv('/home/phack/Documents/Mestrado/Hierarchical_Classifier/breasts2.csv').values
        # data = pd.read_csv('/home/phack/Documents/Mestrado/Hierarchical_Classifier/breasts3.csv').values
        # data = pd.read_csv('/home/phack/Documents/Mestrado/Hierarchical_Classifier/ecoli.csv').values

        # db = load_iris()
        # X = db.data
        db = load_digits()
        X = db.data
        y = db.target

        # X = data[:,1:8]
        # label = data[:,8]
        #
        # classes = unique(label)
        #
        # y = zeros(len(label))
        #
        # for i,l in enumerate(label):
        #
        #     y[i] = where(classes == l)[0]
        #
        # print X.shape
        # y = y.astype(int)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=5)


        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)

        # print 'computing the LDA
        lda = LDA(n_components=10).fit(X_tr, y_tr)
        X_tr = lda.transform(X_tr)
        X_te = lda.transform(X_te)
        # print 'done', X_tr.shape

        # scaler = DynamicClassifier(base_metric=dist.cosine).fit(X_tr, y_tr)
        # X_tr = scaler._transform(X_tr)
        # X_te = scaler._transform(X_te)
        #
        # print X_tr.shape

        # my_d = FastDSM(base_metric=dist.euclidean)#.fit(X_tr, y_tr, verbose=True)

        # model = GradientClassifier(base_distance=p2p_dist,
        #                            beta=0.1,
        #                            alpha=2.,
        #                            repr_method='mean',
        #                            w=2.)
        # model = FactorClassifier(base_metric=dist.matching,
        #                          repr_method='mean',
        #                          alpha=2,
        #                          w=2.,
        #                          beta=0.4)
        # model = DynamicClassifier(dim_transf='none', base_metric=dist.cosine, repr_method='farthest')
        model = HierarchicalClassifier(dim_transf='none', base_metric=dist.matching, repr_method='mean')
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)#, force_class=True)
        # nans = [math.isnan(l) for l in y_pred]
        # nans = where(nans)[0]
        # y_pred[nans] = -1
        # print math.isnan(y_pred[-7])
        # print y_pred

        print classification_report(y_te, y_pred)
        print 'accuracy =', accuracy_score(y_te, y_pred)

        model = SVC().fit(X_tr, y_tr)
        # model = KNeighborsClassifier(1, metric=p2p_dist).fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        print classification_report(y_te, y_pred)
        print 'accuracy =', accuracy_score(y_te, y_pred)


if __name__ == '__main__':
    # loads and runs the Unit Tests
    suite = TestLoader().loadTestsFromTestCase(GradientClassifierTests)
    TextTestRunner(verbosity=2, ).run(suite)