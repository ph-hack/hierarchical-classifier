# DSM = Dynamic Similarity Metric

# imports #######################
from numpy import array, zeros, ones, abs, mean
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import NotFittedError

# core ##########################

class Blob(object):

    def __init__(self):

        self.data = {}

    def __len__(self):

        return len(self.data)

    def __iter__(self):

        return (x for x in self.data)

    def __getitem__(self, item):

        item = self.get_key(item)

        return self.data[str(item)]

    def __setitem__(self, key, value):

        key = self.get_key(key)

        self.data[str(key)] = value

    def __sub__(self, other):

        if not isinstance(other, Blob):

            raise Exception('Subtraction only supports operations between Blobs')

        result = Blob()

        for k in self:

            result[k] = self[k] - other[k]

        return result

    def __abs__(self):

        result = Blob()

        for k in self:

            result[k] = abs(self[k])

        return result

    def __div__(self, other):

        result = Blob()

        if type(other) is int or type(other) is float:

            for k in self:

                result[k] = self[k]/other

            return result

        for k in self:

            result[k] = self[k]/other[k]

        return result

    def __str__(self):

        text = ''

        for k in self:

            text = ''.join([text, '{} = {},\n'.format(k, self[k])])

        return text

    def __repr__(self):

        return '\nBlob: [\n{}]\n'.format(str(self))

    def get_key(self, key):

        if type(key) is list:

            key.sort()
            return key
            # raise Exception('Blob error: list as key is not implemented yet')

        elif type(key) is slice:

            raise Exception('Blob error: slice as key is not implemented yet')

        elif type(key) is float or type(key) is int:

            return [key]

        elif type(key) is str:

            return key

        key = list(key)
        key.sort()

        return key

    def mean(self):

        return array(self.values).mean()

    @property
    def values(self):

        return self.data.values()


class DSM(object):

    def __init__(self, base_metric=euclidean, alpha=0.1, beta=0.1, max_iter=100, max_error=0.1, scaler=None):

        self.f = base_metric
        self.w = None
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.max_error = max_error
        self.scaler = scaler

    def __call__(self, *args, **kwargs):

        if len(args) != 2:

            raise Exception('DSM Error: It takes exactly two vectors as input instead of {}!'.format(len(args)))

        return self.similarity(args[0], args[1])

    def similarity(self, x1, x2):

        try:
            if self.scaler is not None:
                x1 = self.scaler.transform(x1)
                x2 = self.scaler.transform(x2)

        except NotFittedError:

            self.scaler.fit(array([x1, x2]))
            x1 = self.scaler.transform(x1)
            x2 = self.scaler.transform(x2)

        if self.w is None:

            self.w = ones(len(x1))

        return self.f(x1 * self.w, x2 * self.w)

    def fit(self, X, y, verbose=False):

        if self.scaler is not None:
            X = self.scaler.fit(X).transform(X)

        N = X.shape[1]

        tm = DSM.truth_matrix(y)

        # if verbose:
        #     print 'truth-matrix =\n', tm

        self.w = ones(N)

        dm = self.distance_matrix(X)

        error = abs(tm - dm['current']).mean()

        if verbose:
            print 'First error =', error

        temp_error = error

        iteration = 0

        while(error > self.max_error and iteration < self.max_iter and temp_error <= error):

            error = temp_error

            deltas = self.gradient(dm, tm)

            for j in range(len(self.w)):

                self.w[j] -= self.alpha * deltas[j] * self.w[j]

            dm = self.distance_matrix(X)
            temp_error = abs(tm - dm['current']).mean()
            iteration += 1

            if verbose:

                print 'Iteration =', iteration, ' <=> ', self.max_iter
                print 'Error =', temp_error, ' <=> ', self.max_error
                print 'Weights =', self.w

        return self

    def gradient(self, dm, tm):

        gradient = []

        for j in range(len(self.w)):

            Ej = abs(tm - dm['current']).mean()
            Ejb = abs(tm - dm['w{}'.format(j)]).mean()

            gradient.append((Ejb - Ej)/(self.beta))

        return array(gradient)

    def distance_matrix(self, data):

        distances = {}
        N = data.shape[1]

        current_w = self.w.copy()

        distances['current'] = Blob()

        for k in range(N):
            distances['w{}'.format(k)] = Blob()

        for i in range(len(data)-1):

            for j in range(i+1, len(data)):

                distances['current'][i,j] = self.similarity(data[i,:], data[j,:])
                distances['current'][j,i] = distances['current'][i,j]

                for k in range(N):

                    self.w[k] += self.beta

                    distances['w{}'.format(k)][i,j] = self.similarity(data[i,:], data[j,:])
                    distances['w{}'.format(k)][j,i] = distances['w{}'.format(k)][i,j]

                    self.w = current_w.copy()

        return distances

    @staticmethod
    def truth_matrix(labels):

        # errors = zeros((len(labels), len(labels)))
        errors = Blob()

        for i in range(len(labels)-1):

            for j in range(i+1, len(labels)):

                errors[i,j] = 0. if labels[i] == labels[j] else 1.
                errors[j,i] = errors[i,j]

        return errors

class FastDSM(DSM):

    def fit(self, X, y, verbose=False):

        if self.scaler is not None:
            X = self.scaler.fit(X).transform(X)

        N = X.shape[1]

        tm = DSM.truth_matrix(y)

        self.w = ones(N)

        dm = self.distance_matrix(X)

        directions = self.gradient(dm, tm)
        directions /= abs(directions)

        error = abs(tm - dm['current']).mean()

        if verbose:
            print 'First error =', error

        prior_error = error
        prior_w = self.w.copy()
        #TODO: Whether remove these parts of the 'delta'/'gradient' term or make it optional
        # delta = ones(len(self.w))
        # delta = 1.
        # hist_errors = [abs(error - abs(tm - dm['w0']).mean())]

        iteration = 0

        while(error > self.max_error and iteration < self.max_iter and error <= prior_error):
            prior_error = error
            prior_w = self.w.copy()

            for j in range(len(self.w)):

                # self.w[j] -= self.beta * directions[j] * self.alpha * delta[j] #* self.w[j]
                self.w[j] -= self.beta * directions[j] * self.w[j]

            dm = self.distance_matrix(X, only_current=True)
            error = abs(tm - dm['current']).mean()

            # delta = abs(prior_error - error)/abs(prior_w - self.w)
            # delta = abs(prior_error - error)/mean(hist_errors)
            # print 'Delta debug =',abs(prior_error - error), ' / ', abs(prior_w - self.w)
            # print 'Delta debug =',abs(prior_error - error), ' / ', mean(hist_errors)
            # hist_errors.append(abs(error - prior_error))

            iteration += 1

            if verbose:

                print 'Iteration =', iteration, ' <=> ', self.max_iter
                print 'Error =', error, ' <=> ', self.max_error
                print 'Weights =', self.w
                # print 'Deltas =', delta

        if error > prior_error:

            self.w = prior_w

        return self

    def distance_matrix(self, data, only_current=False):

        distances = {}
        N = data.shape[1]

        current_w = self.w.copy()

        distances['current'] = Blob()

        for k in range(N):
            distances['w{}'.format(k)] = Blob()

        for i in range(len(data)-1):

            for j in range(i+1, len(data)):

                distances['current'][i,j] = self.similarity(data[i,:], data[j,:])
                distances['current'][j,i] = distances['current'][i,j]

                if not only_current:
                    for k in range(N):

                        self.w[k] += self.beta

                        distances['w{}'.format(k)][i,j] = self.similarity(data[i,:], data[j,:])
                        distances['w{}'.format(k)][j,i] = distances['w{}'.format(k)][i,j]

                        self.w = current_w.copy()

        return distances

# unit tests ####################

def test_blobs():

    x = Blob()
    x[0,0] = 1.
    x[0,1] = 2.
    x[1,0] = 1.
    x[1,1] = 3.

    print x.data

    y = Blob()
    y[0,0] = 5.
    y[0,1] = 6.
    y[1,0] = 4.
    y[1,1] = 1.

    print y.data

    z = abs(x - y)

    print z.data

    print z.mean()

def test_dsm():

    X = array([[2,3], [5,4], [10,4], [20,3]])
    y = array([0, 0, 1, 1])

    dsm = FastDSM(alpha=1.05, scaler=StandardScaler().fit(X))

    f = dsm.similarity

    print f([2,3], [5,4])

    dm = dsm.distance_matrix(X)

    print 'dm lenght =', len(dm['current'])

    dsm.fit(X, y, True)

    print 'Euclidean =', dm['current']
    dm = dsm.distance_matrix(X)
    print 'DSM Eucliden =', dm['current']


if __name__ == '__main__':

    # test_blobs()
    test_dsm()
    # pass
