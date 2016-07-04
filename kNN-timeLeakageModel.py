import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold


# ## K-Nearest Neighbor Regressor Modelling that avoids Time Leakage

class kNearestNeighborRegressor(object):
    def __init__(self, k=5):
        self.k = k
    
    def euclideanDistance(self, instance1, instance2, date_index=2):
        """
        Compute the euclidean distance between two data instances.
        
        Parameters
        ----------
        instance1 : array-like, shape (1, n_features), \
            Data point 1. 
        instance2 : array-like, shape (1, n_features), \
            Data point 2.
        date_index : int
            Index of first non-numerical feature in dataset
        
        Returns
        -------
        euclid_dst : int
            Euclidean distance between instance1 and instance2
        """
        return np.linalg.norm(instance1[:date_index]-instance2[:date_index])
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def getMRAE(self, y_pred, y_true):
        """
        Compute the Median Relative Absolute Error
        
        Parameters
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values.
        y_pred : 1d array-like, or label indicator array / sparse matrix
            Estimated targets as returned by a classifier.
            
        Returns
        -------
        MRAE: float
            Median Relative Absolute Error of the model
        """
        return np.median(np.abs(y_pred - y_true) / y_true)
        
    def getNeighbors(self, testInstance, n_neighbors=None):
        """
        Finds the K-neighbors of a point while avoiding time leakage.
        Returns indices of the neighbors and distances to the 
        neighbors of each point.
        
        Parameters
        ----------
        testInstance : array-like, shape (1, n_features), \
            The query point.
        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).
        
        Returns
        -------
        distances : array
            Array representing the lengths to points, only present if
            return_distance=True
        neighbors : array
            Indices of the nearest points in the population matrix.        
        """
        if n_neighbors is None:
            n_neighbors = self.k
        distances = []
        list_date_i = testInstance[2]
        for j in range(len(self.X_train)):
            closing_date_j = self.X_train[j][3]
            if closing_date_j < list_date_i: #Prevents Time Leakage
                dist = self.euclideanDistance(testInstance, self.X_train[j])
                distances.append((j, dist))
        distances.sort(key=lambda t: t[1])
        neighbors = []
        if len(distances) >= n_neighbors:
            for j in range(n_neighbors):
                neighbors.append(distances[j][0])
        else:
            for j in range(len(distances)):
                neighbors.append(distances[j][0])
        distances = [t[1] for t in distances[:n_neighbors]]
        return np.array(distances), np.array(neighbors)
    
    def getWeights(self, dist):
        """
        Compute the weighted points of the neighbor data distances
        by the inverse of their distance.
        
        Parameters
        ===========
        dist: ndarray
            The input distances
        
        Returns
        ========
        weights_arr: array of the same shape as ``dist``
            Weighted Points
        """
        for point_dist_i, point_dist in enumerate(dist):
            # check if point_dist is iterable
            # (ex: RadiusNeighborClassifier.predict may set an element of
            # dist to 1e-6 to represent an 'outlier')
            if hasattr(point_dist, '__contains__') and 0. in point_dist:
                dist[point_dist_i] = point_dist == 0.
            else:
                dist[point_dist_i] = 1. / point_dist

        return dist

    def getPrediction(self, testInstance):
        """
        Predict the target for the provided data instance
        
        Parameters
        ----------
        testInstance : array-like, shape (1, n_features), \
            The query point.
        
        Returns
        -------
        y_pred: float
            Target Value
        """
        neigh_dist, neigh_ind = self.getNeighbors(testInstance)
        
        if neigh_ind.size == 0: #Instance when there are no homes_j that are neighbors of home_i due to time rule.
            return np.mean(self.y_train)
        
        weights = self.getWeights(neigh_dist)

        if weights is None:
            y_pred = np.mean(self.y_train[neigh_ind])
        else:
            denom = np.sum(weights)
            num = np.sum(self.y_train[neigh_ind] * weights)
            y_pred = num/denom
        return y_pred
    
    def getPredictions(self, X):
        """
        Predict the target for multiple data instances
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of int, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        """
        y_pred = []

        for j in range(len(X)):
            if j % 1000 == 0:
                print 'Iteration: ', j
            result = self.getPrediction(X[j])
            y_pred.append(result)
        print

        return np.array(y_pred)

if __name__ == '__main__':
    #Loading the Housing Prices Data
    df = pd.read_csv('sample_sales.csv')
    print df.head()
    print 
    print df.info()
    print

    #store response vector in "prices"
    prices = df.pop('close_price').values
    #store feature matrix in "X"
    X = df.values

    #Split Housing prices data Into Training and Test Sets
    X_train, X_test, prices_train, prices_test = train_test_split(X, prices, test_size=0.2, random_state=42)                
    knn = kNearestNeighborRegressor(k=4)
    knn.fit(X_train, prices_train)

    # ## Model Performance

    prices_pred = knn.getPredictions(X_test)
    print 'The Median Relative Absolute Error of our KNN Regressor is: %s' %knn.getMRAE(prices_pred, prices_test)
    print

    # ### Estimating Median Relative Absolute Error with Cross-Validation 
    kf = KFold(len(X), n_folds=6)
    fold = 1
    scores = []
    for train_index, test_index in kf:
        print "Fold #%s" %fold
        Xf_train, Xf_test = X[train_index], X[test_index]
        prices_f_train, prices_f_test = prices[train_index], prices[test_index]
        knn_f = kNearestNeighborRegressor(k=4)
        knn_f.fit(Xf_train, prices_f_train)
        prices_f_pred = knn_f.getPredictions(Xf_test)
        scores.append(knn.getMRAE(prices_f_pred, prices_f_test))
        fold += 1
    scores = np.array(scores)
    print "MRAE: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2)
