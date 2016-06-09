import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.lda import LDA
from numpy import mean, zeros, where, unique, maximum, array
from scipy.spatial.distance import euclidean, cosine


def distance_matrix(data, metric=euclidean):

    distances = zeros((len(data), len(data)))

    for i in range(len(data)-1):

        for j in range(i+1, len(data)):

            distances[i,j] = metric(data[i,:], data[j,:])
            distances[j,i] = distances[i,j]

    return distances

def cluster_quality(data, labels, metric=euclidean):

    s_labels = unique(labels)

    if metric == 'precomputed':

        distances = data

    else:
        distances = distance_matrix(data, metric)

    intra_distances = []
    inter_distances = []

    outlier_prop = len(where(labels == -1)[0].tolist())/float(len(data))

    for l in s_labels:

        if l != -1:

            in_samples = where(labels == l)[0].tolist()
            in_distances = [distances[i,list(set(in_samples) - set([i]))].mean() for i in in_samples]
            intra_distances.append(mean(in_distances))

            out_samples = where(labels != l)[0].tolist()

            if len(out_samples) > 0:

                out_distances = [distances[i, out_samples].mean() for i in in_samples]

            else:

                return 0.

            inter_distances.append(mean(out_distances))

    inter = mean(inter_distances)
    intra = mean(intra_distances)

    return (inter - intra)/(maximum(inter, intra)) - outlier_prop

def remove_repetition(data):

    new_data = []
    added = []

    for i, d in enumerate(data):

        if d.tolist() not in new_data:

            new_data.append(d.tolist())
            added.append(i)

    return array(new_data), added

class SmartDBSCAN(object):

    def __init__(self, metric=euclidean):

        self.metric = metric
        self.model = None

    def fit(self, X, eps, min_samples, verbose=False):

        dm_array = distance_matrix(X, self.metric)
        dm = dm_array.tolist()

        if verbose:

            figs, axes = plt.subplots(1,2)
            axes[0].imshow(dm_array, interpolation='none')
            im = axes[0].pcolormesh(dm_array)
            figs.colorbar(im, ax=axes[0])
            axes[0].set_title('distances')

            axes[1].hist(dm_array.flat)
            axes[1].set_title('distances distribution')
            plt.show()

        chosen_param = {'eps': 0, 'min_samples': 0}
        temp_score = 0.

        for e in eps:
            for s in min_samples:

                model = DBSCAN(e, s, 'precomputed').fit(dm)
                labels = model.labels_
                score = cluster_quality(dm_array, labels, metric='precomputed')

                if verbose:
                    print 'd =', e, ', s =', s, ', ->', score, ', clusters =', len(unique(labels))

                if score > temp_score:

                    self.model = model
                    temp_score = score

    @property
    def labels_(self):

        return self.model.labels_

def test():

    db = load_digits(3)
    X = db.data
    X = StandardScaler().fit(X).transform(X)
    labels = db.target
    pca = LDA(n_components=2)
    data = pca.fit(db.data, db.target).transform(db.data)
    X, uniques = remove_repetition(X)

    print db.data.shape

    distances = [0.05, 0.1, 0.15, 0.2, 0.23, 0.25]
    n_samples = [3, 4, 5, 7, 10, 15]

    metric = cosine

    model = SmartDBSCAN(metric=metric)
    model.fit(data, distances, n_samples, verbose=True)

    pred = model.labels_

    c_tags = ['ro', 'bo', 'go', 'yo', 'm*', 'c*', 'w*']

    _, axes = plt.subplots(1,2)

    for p in pred:
        axes[0].plot(data[where(pred == p)[0],0], data[where(pred == p)[0],1], c_tags[p])

    axes[0].plot(data[where(pred == -1)[0],0], data[where(pred == -1)[0],1], 'k*')

    for p in labels:
        axes[1].plot(data[where(labels == p)[0],0], data[where(labels == p)[0],1], c_tags[p])

    plt.show()

if __name__ == '__main__':

    test()