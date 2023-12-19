from copy import deepcopy

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import auc, precision_recall_curve

import decomposition

mpl.use('Qt5Agg')

# Does binary classification.
class Classifier(BaseEstimator, ClassifierMixin):
    # Initializes the Classifier.
    def __init__(self, k, alpha, X):
        # hyperparameters
        self.k = k
        self.alpha = alpha

        # dataframe indexed with chems and with ses as columns, filled with 0's
        self.X = X

    # Fits the classifier to the given data.
    def fit(self, nodes, y=None):
        self.X = pd.DataFrame(int(0), index=self.X.index, columns=self.X.columns)

        # set values for training samples
        for i in nodes.index:
            chem = nodes['source'][i]
            se = nodes['target'][i]
            type = nodes['type'][i]
            self.X[se][chem] = type

        # run decomposition algorithm
        self.W_, self.H_, self.J_, self.deltas_ = decomposition.DecompositionAlgorithm(self.X.to_numpy(), self.k,
                                                                                       self.alpha)
        R_hat_ = np.matmul(self.W_, self.H_)
        self.nodes_hat_ = pd.DataFrame(R_hat_, columns=self.X.columns, index=self.X.index)

        return self

    # Gives predictions according to the fitted classifier.
    def _meaning(self, chem, se):
        # score of chem-se pair
        score = self.nodes_hat_[se][chem]

        return score


    # Looks up the meaning for all chem-se pairs in X.
    def predict(self, X, y=None):
        try:
            getattr(self, "nodes_hat_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        pred_matrix = deepcopy(X)

        preds = [self._meaning(pred_matrix['source'][idx] , pred_matrix['target'][idx]) for idx in X.index]
        return preds

    # Scores the fitted classifier on data given in X.
    # Returns the PR AUC.
    def score(self, X, y=None):
        # get predictions and true scores
        pred = self.predict(X)
        true = [x > 0 for x in X.type.to_list()]

        # calculate pr-auc score
        precision, recall, thresholds = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)

        return pr_auc