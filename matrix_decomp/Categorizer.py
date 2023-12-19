from copy import deepcopy

import pandas as pd
import sklearn.metrics
from sklearn.base import BaseEstimator, ClassifierMixin
import decomposition
import numpy as np

# Does categorical classification.
class Categorizer(BaseEstimator, ClassifierMixin):
    # Initializes the categorizer.
    def __init__(self, k, alpha, X, thresholds=None, classes=None):
        # hyperparameters
        self.k = k
        self.alpha = alpha

        # dataframe indexed with chems and with ses as columns, filled with 0's
        self.X = X

        # thresholds at which maximum likelihood estimate changes and corresponding class estimates
        self.thresholds = thresholds
        self.classes = classes

    # Fits the categorizer to the given data.
    def fit(self, nodes, y=None):
        self.X = pd.DataFrame(int(0), index=self.X.index, columns=self.X.columns)

        # set values for training samples
        for i in nodes.index:
            chem = nodes['source'][i]
            se = nodes['target'][i]
            type = nodes['type'][i]
            self.X[se][chem] = type

        # run decomposition algorithm
        self.W_, self.H_, self.J_, self.deltas_ = decomposition.DecompositionAlgorithm(self.X.to_numpy(), self.k, self.alpha)
        R_hat_ = np.matmul(self.W_, self.H_)
        self.nodes_hat_ = pd.DataFrame(R_hat_, columns=self.X.columns, index=self.X.index)

        return self

    # Gives predictions according to the fitted categorizer.
    def _meaning(self, chem, se):
        # score of chem-se pair
        score = self.nodes_hat_[se][chem]

        # if threshold is given output corresponding class
        if self.thresholds is not None:
            if score < self.thresholds[0]:
                return self.classes[0]
            for i in range(1, len(self.thresholds)):
                if score <= self.thresholds[i] and score > self.thresholds[i-1]:
                    return self.classes[i]
            return self.classes[-1]
        # else use nearest class
        else:
            if score > 5:
                return 5
            else:
                return round(score)


    # Looks up the meaning for all chem-se pairs in X.
    def predict(self, X, y=None):
        try:
            getattr(self, "nodes_hat_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        pred_matrix = deepcopy(X)

        return([self._meaning(pred_matrix['source'][idx] , pred_matrix['target'][idx]) for idx in X.index])


    # Scores the fitted categorier on data given in X.
    # Returns the f1-score.
    def score(self, X, y=None, perClass = False):
        # get predictions and true scores
        pred = self.predict(X)
        true = [int(x) for x in X.type.to_list()]

        # return unweighted mean of f1-scores per class
        score = sklearn.metrics.f1_score(y_true=true, y_pred=pred, average='macro')
        return score