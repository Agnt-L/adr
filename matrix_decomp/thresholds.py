from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import Regressor
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')
import scipy.stats as st

# Writes a_list to file with given name.
def write_list(a_list, name):
    print("Started writing list data into a json file")
    with open(name, "w") as fp:
        json.dump(a_list, fp)
        print("Done writing JSON data into .json file")

# Reads list in file with given name to memory.
def read_list(name):
    # open file in read binary mode
    with open(name, 'rb') as fp:
        n_list = json.load(fp)
    # return list object
    return n_list

# Creates thresholds and corresponding classes for seperating frequency classes 1 to 5.
def make_thresholds(train, zeros, k, alpha, cv=1, cv_testsize=0.2):
    # collect true and predicted classes
    true = []
    preds = []
    for i in range(cv):
        tr, te = train_test_split(train, test_size=cv_testsize)

        # fit Regressor to get prediction scores
        regr = Regressor.Regressor(k, alpha, zeros)
        regr.fit(tr)

        preds += regr.predict(te)
        true += te.type.to_list()

    # array of gaussian kernel functions for each class
    gaussians = []
    # array of values in each class
    vals_len = []

    # x-axis of plot
    x = np.arange(0, 6, 0.001)
    # iterate over all classes
    for j in range(6):
        # collect predicted values for actual class j
        vals = [preds[i] for i in range(len(true)) if true[i] == j]
        # plot a histogram of these values
        plt.hist(vals, bins=30, density=True, label="Data")
        # fit a gaussion kernel
        kde = st.gaussian_kde(vals, bw_method=0.5)
        # append it to array
        gaussians.append(kde)
        # append number of values in class j
        vals_len.append(len(vals))
        # plot gaussian kernel function for class j
        plt.plot(x, kde.pdf(x), '-')

    # find best threshold for binary classification weather
    # drug-se pair has 0 frequency or not
    f1s = []
    accs = []
    mccs = []
    zero_threshs = np.arange(0, 6, 0.001)
    for t in zero_threshs:
        class0_true = [x == 0 for x in true]
        class0_pred = [x <= t for x in preds]
        f1 = f1_score(y_true=class0_true, y_pred=class0_pred)
        acc = accuracy_score(y_true=class0_true, y_pred=class0_pred)
        mcc = matthews_corrcoef(y_true=class0_true, y_pred=class0_pred)
        f1s.append(f1)
        accs.append(acc)
        mccs.append(mcc)
    best_thresh = [zero_threshs[np.argmax(f1s)], zero_threshs[np.argmax(accs)], zero_threshs[np.argmax(mccs)]]

    # get probability density functions for gaussian kernel of each class
    pdfs = [gaussians[i].pdf(x) for i in range(len(gaussians))]
    # transpose pdf array
    t_pdfs = np.array(pdfs).T.tolist()
    # for each x value get the maximum pdf y value
    pdfs_max = np.array([max(x) for x in t_pdfs])
    # get the class that the highest y value belongs to
    pdfs_max_idx = [x.index(max(x)) for x in t_pdfs]
    # get the indices at which the mle changes
    idx = np.argwhere(np.diff(pdfs_max_idx)).flatten()
    # get score values that correspond to mle changes
    thresholds = x[idx].tolist()
    # plot the thresholds
    plt.plot(thresholds, pdfs_max[idx], 'ro')

    # plot zero threshold calculated from binary f1-scores
    plt.axvline(best_thresh[0], c='r')
    plt.axvline(best_thresh[1], c='r', linestyle='--')
    plt.axvline(best_thresh[2], c='r', linestyle='-.')

    plt.savefig('data/threshold_plot')
    plt.close()

    # Make list of classes to be predicted when score is
    # below / over a certain threshold. Classes list is always
    # one entry longer than thresholds list.
    classes = np.array(pdfs_max_idx)[idx].tolist()
    classes1 = np.array(pdfs_max_idx)[idx+1].tolist()
    classes.append(classes1[-1])

    # Write thresholds and classes lists with zero class threshold
    # being predicted using gaussian kernels like for the other classes.
    write_list(thresholds, 'data/thresholds.json')
    write_list(classes, 'data/classes.json')

    # Make new lists with the thresholds for zero class calculated
    # using f1-scores.
    classes[0] = 0
    thresholds[0] = best_thresh[0]

    write_list(thresholds, 'data/thresholds_zero.json')
    write_list(classes, 'data/classes_zero.json')

    write_list(best_thresh, 'data/zero.json')

# find best threshold for binary classification determining weather
# a drug-se pair has 0 frequency or not
def binary_thresh(train, zeros, k, alpha, cv=1, cv_testsize=0.2):
    # collect true and predicted classes
    true = []
    preds = []
    for i in range(cv):
        tr, te = train_test_split(train, test_size=cv_testsize)

        # fit Regressor to get prediction scores
        regr = Regressor.Regressor(k, alpha, zeros)
        regr.fit(tr)

        preds += regr.predict(te)
        true += te.type.to_list()

    # metrics to calculate
    scorers = {
        'f1_score': f1_score,
        'acc': accuracy_score,
        'mcc': matthews_corrcoef
    }

    best_threshs = []
    zero_threshs = np.arange(0, 6, 0.001)

    # iterate over metrics
    for scorer in scorers:
        scores = []
        # iterate over thresholds
        for t in zero_threshs:
            class1_true = [x > 0 for x in true]
            class1_pred = [x >= t for x in preds]
            score = scorers[scorer](y_true=class1_true, y_pred=class1_pred)
            scores.append(score)
        # get best score and corresponding threshold
        max_idx = np.argmax(scores)
        best_thresh = zero_threshs[max_idx]
        # save best threshold
        best_threshs.append(best_thresh)
    return best_threshs
