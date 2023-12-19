import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Qt5Agg')
import numpy as np
import oapackage
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

import Categorizer
import Classifier
import Regressor
import neo_graph
import thresholds
import evaluation

# default decomposition parameters
k = 14
alpha = 0.5

# Performs grid search on same set of hyperparameters for classification and regression
# and returns the search objects.
def gridRegClf(params, cv, zeros, train):
    clf = Classifier.Classifier(k, alpha, zeros)
    clfSearch = GridSearchCV(estimator=clf, param_grid=params, cv=cv)
    clfSearch.fit(train)

    reg = Regressor.Regressor(k, alpha, zeros)
    regSearch = GridSearchCV(estimator=reg, param_grid=params, cv=cv, error_score='raise')
    regSearch.fit(train)
    return clfSearch, regSearch

# Makes a contour plot of the results of the given search object.
def plot_results(search):
    plt.figure()

    params = search.param_grid

    X = params['alpha']
    Y = params['k']
    Z = [search.cv_results_['mean_test_score'].tolist()[i * len(params['alpha']):(i + 1) * len(params['alpha'])] for
         i in range(len(params['k']))]

    plt.contourf(X, Y, Z)

# Receives a list of dictionaries each containing a hyperparameter combination.
# Creates a list of dicts only containing one value per hyperparameter
# to make sure all given hyperparameter combinations are sampled once.
def paramDictList(sampled_params):
    clf_dicts = []
    for dict in sampled_params:
        clf_dict = {}
        for k, v in dict.items():
            clf_dict.update({k: [v]})
        clf_dicts.append(clf_dict)

    return clf_dicts

# Returns the pareto optimal hyperparameter configurations of the 2 given search objects
# meaning that all configurations are omitted for which there is another configuration that has a better score according
# to both search objects.
def pareoOpt(search1, search2):
    points1 = search1.cv_results_['mean_test_score'].tolist()
    points2 = search2.cv_results_['mean_test_score'].tolist()

    pareto=oapackage.ParetoDoubleLong()

    for ii in range(0, len(points1)):
        w=oapackage.doubleVector( (points1[ii], -points2[ii]))
        pareto.addvalue(w, ii)

    pareto.show(verbose=1)

    lst = list(pareto.allindices())  # the indices of the Pareto optimal designs
    print('datapoints begin')
    print(points1)
    print(points2)
    print('datapoints end')
    optimal_datapoints = [search1.cv_results_['params'][x] for x in lst]
    s1_scores = [search1.cv_results_['mean_test_score'][x] for x in lst]
    s2_scores = [search2.cv_results_['mean_test_score'][x] for x in lst]
    return optimal_datapoints, s1_scores, s2_scores


params = {"alpha": np.linspace(0.2, 0.5, 5), "k": range(10, 15)}

# Performs grid search on Classifier and Regressor and prints pareto optimal hyperparameter configurations
# along with their binary classification and regression scores.
def gridSearchExp(cv, zeros, train):
    clfSearch, regSearch = gridRegClf(params, cv, zeros, train)
    plot_results(clfSearch)
    plot_results(regSearch)
    optimal_datapoints, clf_scores, reg_scores = pareoOpt(clfSearch, regSearch)
    print(optimal_datapoints)
    print(clf_scores)
    print(reg_scores)

# Performs hyperparameter search for regressor and binary classifier on the same random hyperparameter configurations
# and prints pareto optimal configurations along with their binary and regression scores.
def randSearchExp(cv, n_iter, zeros, train):
    reg = Regressor.Regressor(k, alpha, zeros)
    regSearch = RandomizedSearchCV(estimator=reg, param_distributions=params, cv=cv, error_score='raise', n_iter=n_iter)
    regSearch.fit(train)

    clf = Classifier.Classifier(k, alpha, zeros)
    clf_dicts = paramDictList(regSearch.cv_results_['params'])
    clfSearch = RandomizedSearchCV(estimator=clf, param_distributions=clf_dicts, cv=cv, error_score='raise',
                                   n_iter=n_iter)
    clfSearch.fit(train)

    optimal_datapoints, clf_scores, reg_scores = pareoOpt(clfSearch, regSearch)
    with open('data/pareto_opt_scores', 'w') as f:
        f.write(str(optimal_datapoints) + '\n')
        f.write(str(clf_scores) + '\n')
        f.write(str(reg_scores) + '\n')
        f.write(str(clf_dicts))


# Performs categorical evaluation for given true and predicted values and saves the results to a file named [name].
def clf_evaluation_cat(y_true, pred, name):
    # Returns the accuracies per class between true values and predicted values as an array
    # starting with the ACC for class 0.
    def acc_per_class(true, pred):
        accs = []
        for c in range(6):
            true_c = [x==c for x in true]
            pred_c = [x==c for x in pred]
            class_acc = accuracy_score(true_c, pred_c)
            accs.append(class_acc)
        return accs

    # Returns the MCCs per class between true values and predicted values as an array
    # starting with the MCC for class 0.
    def mcc_per_class(true, pred):
        mccs = []
        for c in range(6):
            true_c = [x==c for x in true]
            pred_c = [x==c for x in pred]
            class_acc = matthews_corrcoef(true_c, pred_c)
            mccs.append(class_acc)
        return mccs

    # define metrics to be calculated
    scorers = {
        'f1-scores per class': lambda x, y: f1_score(x, y, average=None),
        'unweighted mean of f1-scores per class': lambda x, y : f1_score(x, y, average='macro'),
        'acc': accuracy_score,
        'acc per class': acc_per_class,
        'mcc': matthews_corrcoef,
        'mcc per class': mcc_per_class
    }

    # round predictions to nearest class, only makes a difference when no thresholds are applied
    y_pred = [round(x) for x in pred]

    # write results to evaluation file
    with open(name, 'w') as f:
        f.write('\nconf matrix:')
        f.write('\n' + str(confusion_matrix(y_true, y_pred)))
        f.write('\n')
        j = 0
        for scorer in scorers:
            f.write(f'\n{scorer}: {scorers[scorer](y_true, y_pred)}')
            f.write('\n')
            j += 1

    # count also predictions for one of the neighboring classes as a correct prediction
    for i in range(len(y_pred)):
        if abs(y_true[i] - y_pred[i]) <= 1:
            y_pred[i] = y_true[i]

    # write new results to evaluation file
    with open(name, 'a') as f:
        f.write('\n')
        f.write("\nexact or neighbor class")
        f.write('\n' + str(confusion_matrix(y_true, y_pred)))
        f.write('\n')
        j = 0
        for scorer in scorers:
            f.write(f'\n{scorer}: {scorers[scorer](y_true, y_pred)}')
            f.write('\nconf matrix:')
            f.write('\n')
            j += 1


# Does binary evaluation of dummy and binary classifier.
def final_eval_bin(clf_test, test, dummy, bin_thresh, ext):
    # get test labels as a list
    y_test = test["type"].values.tolist()

    # make binary test set for binary evaluation
    y_test_bin = [x > 0 for x in y_test]

    # make dummy predictions
    dummy_pred = dummy.predict(y_test_bin)

    # binary evaluation
    evaluation.binary_evaluation_final_pred(y_test_bin, dummy_pred, 'mat_dec_binary_evaluation_dummy' + ext)
    evaluation.binary_evaluation_final(y_test_bin, clf_test.predict(test), 'mat_dec_binary_evaluation' + ext,
                                       'Matrix Decomposition', bin_thresh)

# Does categorical evaluation of dummy and binary classifier.
def final_eval_cat(clf_test, test, dummy, prefix):
    # get test labels as a list
    y_test = test["type"].values.tolist()

    # make dummy predictions
    dummy_pred = dummy.predict(y_test)

    # classifier predictions
    pred = clf_test.predict(test)

    # save evaluation
    clf_evaluation_cat(y_test, dummy_pred, prefix + 'categroization_results_dummy')
    clf_evaluation_cat(y_test, pred, prefix + 'categroization_results')

    regression_evaluation(pred, y_test, 'cat_results_mse.txt')


# Performs final evaluation of regression model using the given hyperparameters k and alpha
# on given train and test set.
def final_eval_regression(k, alpha, zeros, train, test, ext):
    # training
    reg_test = Regressor.Regressor(k, alpha, zeros)
    reg_test.fit(train)

    # plot training curves
    plot_loss(reg_test, 'mat-dec_tr-curve-regression_loss' + ext)
    plot_loss(reg_test, 'mat-dec_tr-curve-regression_loss_zoom' + ext, 20)
    plot_deltas(reg_test, 'mat-dec_tr-curve-regression_deltas' + ext)
    plot_deltas(reg_test, 'mat-dec_tr-curve-regression_deltas_zoom' + ext, 0.75)

    # regressor evaluation
    y_pred = reg_test.predict(test)
    y_true = test.type.to_list()

    regression_evaluation(y_pred, y_true, 'regression_results' + ext)


# Calculates regression evaluation metrics.
def regression_evaluation(y_pred, y_true, name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)

    with open('data/' + name, 'w') as f:
        f.write(f'rmse: {rmse}, standardized rmse: {rmse / np.std(y_true)}\n')
        f.write(f'mse: {mse}, standardized mse: {mse / np.std(y_true)}\n')
        f.write(f'mae: {mae}, standardized mae: {mae / np.std(y_true)}\n')


# Performs final evaluation of categorical classification model using the given
# hyperparameters k and alpha on given train and test set.
def final_eval_categorization(k, alpha, zeros, train, test):
    # prepare training data for binary dummy classifier
    X_train = train[["source", "target"]].values.tolist()
    y_train = train["type"].values.tolist()

    # make dummy categorizer that predicts each class with probability equal to its frequency
    dummy_cat = DummyClassifier(strategy='stratified')
    dummy_cat.fit(X_train, y_train)

    # make thresholds for categorizing between frequency classes
    thresholds.make_thresholds(train, zeros, k, alpha, 3)

    # normal training
    cat_test = Categorizer.Categorizer(k, alpha, zeros)
    cat_test.fit(train)

    # plot training curves
    plot_loss(cat_test, 'mat-dec_tr-curve-categorization_loss')
    plot_loss(cat_test, 'mat-dec_tr-curve-categorization_loss_zoom', 20)
    plot_deltas(cat_test, 'mat-dec_tr-curve-categorization_deltas')
    plot_deltas(cat_test, 'mat-dec_tr-curve-categorization_deltas_zoom', 0.75)

    # categorical evaluation without thresholds
    final_eval_cat(cat_test, test, dummy_cat, "data/no_treshs_")

    cat_test.thresholds = thresholds.read_list('data/thresholds.json')
    cat_test.classes = thresholds.read_list('data/classes.json')

    # categorical evaluation with thresholds
    final_eval_cat(cat_test, test, dummy_cat, "data/treshs_")

    cat_test.thresholds = thresholds.read_list('data/thresholds_zero.json')
    cat_test.classes = thresholds.read_list('data/classes_zero.json')

    # categorical evaluation with thresholds and zero threshold
    # being estimated using binary f1-scores
    final_eval_cat(cat_test, test, dummy_cat, "data/treshs_zero_")

# Plots a curve showing the loss of the given predictor over all iterations and saves it to data/[name]
# y_limit can be used to restrict the length of the y-axis.
def plot_loss(predictor, name, y_limit=None):
    plt.figure()
    plt.title('loss curve - binary')
    if y_limit is not None:
        plt.ylim(0, y_limit)  # Set y-axis limits
    plt.plot(range(len(predictor.J_)), predictor.J_)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    # show the legend
    plt.legend()
    plt.savefig('data/' + name)
    plt.close()

# Plots a curve showing the maximum change in factors for the given predictor over all iterations
# and saves it to data/[name]
# y_limit can be used to restrict the length of the y-axis.
def plot_deltas(predictor, name, y_limit=None):
    plt.figure()
    plt.title('max change in factors curve - binary')
    if y_limit is not None:
        plt.ylim(0, y_limit)  # Set y-axis limits
    plt.plot(range(len(predictor.deltas_)), predictor.deltas_)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    # show the legend
    plt.legend()
    plt.savefig('data/' + name)
    plt.close()

# Performs final evaluation of binary classification model using the given
# hyperparameters k and alpha on given train and test set.
def final_eval_binary(k, alpha, zeros, train, test, bin_thresh, ext):
    # prepare training data for binary dummy classifier
    X_train = train[["source", "target"]].values.tolist()
    y_train = train["type"].values.tolist()

    # make dummy binary classifier that always predicts most frequent class
    y_train = [x > 0 for x in y_train]
    dummy_bin = DummyClassifier(strategy='stratified')
    dummy_bin.fit(X_train, y_train)


    # normal training
    clf_test = Classifier.Classifier(k, alpha, zeros)
    clf_test.fit(train)
    plot_loss(clf_test, 'mat-dec_tr-curve-binary_loss' + ext)
    plot_loss(clf_test, 'mat-dec_tr-curve-binary_loss_zoom' + ext, 20)
    plot_deltas(clf_test, 'mat-dec_tr-curve-binary_deltas' + ext)
    plot_deltas(clf_test, 'mat-dec_tr-curve-binary_deltas_zoom' + ext, 0.75)

    # binary evaluation
    final_eval_bin(clf_test, test, dummy_bin, bin_thresh, ext)

def main():
    # retrieve nodes with frequency classes or with continuous frequencies respectively
    nodes_classes, _ = neo_graph.pos_freq_nodes_from_neo4j(True)
    nodes_cont, matrix = neo_graph.pos_freq_nodes_from_neo4j(False)

    # plot frequencies
    neo_graph.plot_freq_hist(nodes_classes['type'].values, 'data/mat_dec_freq_hist_classes.png')
    neo_graph.plot_freq_hist([int(x == 0) for x in nodes_classes['type'].values.tolist()],
                             'data/mat_dec_freq_hist_bin.png')
    neo_graph.plot_freq_hist(nodes_cont['type'].values, 'data/mat_dec_freq_hist_cont.png')

    # make train test split
    train_classes, test_classes = train_test_split(nodes_classes, test_size=0.3)
    train_cont, test_cont = train_test_split(nodes_cont, test_size=0.3)

    pos_train = len([x for x in train_cont['type'].values.tolist() if x > 0])
    neg_train = len([x for x in train_cont['type'].values.tolist() if x == 0])

    pos_test = len([x for x in test_cont['type'].values.tolist() if x > 0])
    neg_test = len([x for x in test_cont['type'].values.tolist() if x == 0])

    # save data set sizes
    with open('data/sizes', 'w') as f:
        f.write(f'train\n')
        f.write(f'pos: {pos_train}, neg: {neg_train}\n')
        f.write(f'test\n')
        f.write(f'pos: {pos_test}, neg: {neg_test}\n')

    # plot frequencies
    neo_graph.plot_freq_hist(test_classes['type'].values, 'data/mat_dec_freq_hist_classes_test.png')
    neo_graph.plot_freq_hist([int(x == 0) for x in test_classes['type'].values.tolist()],
                             'data/mat_dec_freq_hist_bin_test.png')
    neo_graph.plot_freq_hist(test_cont['type'].values, 'data/mat_dec_freq_hist_cont_test.png')


    # create zero filled dataframe to initialize estimators with chems and ses
    zeros = pd.DataFrame(int(0), index=matrix.index, columns=matrix.columns)

    # Hyperparameter optimization
    # One has to choose one of the outputted pareto optimal hyperparameter configurations and set the global variables
    # alpha and k to them.
    randSearchExp(5, 10, zeros, train_cont)

    # Final evaluation
    final_eval_regression(k, alpha, zeros, train_classes, test_classes, '_classes')
    final_eval_regression(k, alpha, zeros, train_cont, train_cont, '_cont')
    final_eval_categorization(k, alpha, zeros, train_classes, test_classes)
    bin_tresh = thresholds.read_list('data/zero.json')
    final_eval_binary(k, alpha, zeros, train_classes, test_classes, bin_tresh, '_classes')
    bin_tresh2 = thresholds.binary_thresh(train_cont, zeros, k, alpha, 3)
    final_eval_binary(k, alpha, zeros, train_cont, test_cont, bin_tresh2, '_cont')

main()