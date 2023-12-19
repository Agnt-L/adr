import autograd.numpy.random as npr
import matplotlib.pyplot as plt
import numpy as np
from autograd import grad
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from neuralfingerprint import (build_morgan_deep_net, build_conv_deep_net,
                               normalize_array, adam, build_batched_grad,
                               mean_squared_error, build_mean_predictor)
from neuralfingerprint.build_vanilla_net import sigmoid

# Builds a predictor based on the given hyperparameters and the net_type. Morgan means that morgan fingerprints are
# used. Conv means that neural fingerprints are created. Linear means that the classification on top of the
# fingerprints is done with linear regression. Net means that the classification is done using a final layer of neural
# network.
def build_predictor(net_type, fp_length, fp_depth, conv_width, h1_size, L2_reg, nll_func):
    if net_type == 'mean':
        return build_mean_predictor(nll_func)
    elif net_type == 'conv_plus_linear':
        vanilla_net_params = dict(layer_sizes=[fp_length],
                                  normalize=True, L2_reg=L2_reg, nll_func=nll_func)
        conv_params = dict(num_hidden_features=[conv_width] * fp_depth,
                           fp_length=fp_length)
        return build_conv_deep_net(conv_params, vanilla_net_params)
    elif net_type == 'morgan_plus_linear':
        vanilla_net_params = dict(layer_sizes=[fp_length],
                                  normalize=True, L2_reg=L2_reg, nll_func=nll_func)
        return build_morgan_deep_net(fp_length, fp_depth, vanilla_net_params)
    elif net_type == 'conv_plus_net':
        vanilla_net_params = dict(layer_sizes=[fp_length, h1_size],
                                  normalize=True, L2_reg=L2_reg, nll_func=nll_func)
        conv_params = dict(num_hidden_features=[conv_width] * fp_depth,
                           fp_length=fp_length)
        return build_conv_deep_net(conv_params, vanilla_net_params)
    elif net_type == 'morgan_plus_net':
        vanilla_net_params = dict(layer_sizes=[fp_length, h1_size],
                                  normalize=True, L2_reg=L2_reg, nll_func=nll_func)
        return build_morgan_deep_net(fp_length, fp_depth, vanilla_net_params)
    else:
        raise Exception("Unknown network type.")

# Trains the neural network.
def train_nn(net_objects, smiles, raw_targets, callback, normalize_outputs,
             seed, init_scale, batch_size, num_iters, **opt_params):
    # get loss function, predictor function and parser
    loss_fun, pred_fun, net_parser = net_objects
    # initialize the weights with random samples from a normal distribution of mean 0 and variance 1
    # , multiplied by init_scale
    init_weights = init_scale * npr.RandomState(seed).randn(len(net_parser))
    # if outputs should be normalized, do that and save the inverse function
    if normalize_outputs:
        targets, undo_norm = normalize_array(raw_targets)
    else:
        targets, undo_norm = raw_targets, lambda x: x

    def make_predict_func(new_weights):
        return lambda new_smiles: undo_norm(pred_fun(new_weights, new_smiles))

    def opt_callback(weights, i):
        callback(make_predict_func(weights), i)


    # function to compute the gradient of the loss function at the current value
    grad_fun = build_batched_grad(grad(loss_fun), batch_size, smiles, targets)
    # get the trained weights by repeatedly calculating the gradient and adjusting
    # the weights in this direction
    trained_weights = adam(grad_fun, init_weights, callback=opt_callback,
                           num_iters=num_iters, **opt_params)

    return trained_weights, make_predict_func(trained_weights)


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params, fp_length, fp_depth, conv_width, h1_size, L2_reg, init_scale, step_size, nll_func):
        """
        Called when initializing the classifier
        """
        self.params = params

        # model params
        self.fp_length = fp_length
        self.fp_depth = fp_depth
        self.conv_width = conv_width  # conv net only
        self.h1_size = h1_size
        self.L2_reg = L2_reg

        # train params
        self.init_scale = init_scale
        self.step_size = step_size

        self.nll_func = nll_func


    # Fits the classifier to the given data.
    def fit(self, X_train, y_train):
        # make test size a multiple of batch_size
        batch_size = self.params['train']['batch_size']
        test_size = int(np.floor((0.2 * len(X_train)) / batch_size) * batch_size)

        # perform split into training and test set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= test_size)

        # build the neural nets
        net_objects = build_predictor(self.params['model']['net_type'], self.fp_length,
                                      self.fp_depth, self.conv_width, self.h1_size, self.L2_reg, self.nll_func)

        # function to compute the negative log likelihood
        def compute_nll(predictor, inputs, targets):
            return self.nll_func(predictor(inputs), targets)

        num_iters, num_records = self.params['train']['num_iters'], self.params['num_records']
        record_idxs = set(map(int, np.linspace(num_iters - 1, 0, num_records)))

        self.training_curve = []
        self.valid_curve = []
        self.record_idxs = []

        # report the nll at particular indices
        def callback(predictor, i):
            if i in record_idxs:
                self.record_idxs.append(i)

                # print current training and validation loss
                print(compute_nll(predictor, X_train, y_train))
                print(compute_nll(predictor, X_val, y_val))

                # append current training and validation loss to curves
                self.training_curve.append(compute_nll(predictor, X_train, y_train))
                self.valid_curve.append(compute_nll(predictor, X_val, y_val))

        train_dict = self.params['train']

        # set trained weights and predictor function as attributes of the classifier
        self.weights_, self.predictor_ =  train_nn(net_objects, X_train, y_train, callback,
                 (self.nll_func == mean_squared_error), train_dict['seed'], self.init_scale,  train_dict['batch_size'],
                                                  train_dict['num_iters'])

        return self

    # Plots training and validation curves of the trained classifier.
    def plot_train_val_curves(self, name):
        plt.plot(self.record_idxs, self.training_curve, label='nll on training set')
        plt.plot(self.record_idxs, self.valid_curve, label='nll on validation set')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(name)
        plt.close()

    # Gives 0 / 1 prediction based on predicted score x.
    def meaning(self, x):
        sig_pred = sigmoid(x)
        bin_pred = int(sig_pred > 0.5)
        return bin_pred

    # Predicts scores for given smiles array X.
    def predict(self, X,):
        try:
            getattr(self, "weights_")
            getattr(self, "predictor_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return self.predictor_(X)

    # Returns f1-score based on given smiles array X and true values y.
    def score(self, X, y):
        preds = [self.meaning(x) for x in self.predict(X)]
        score = f1_score(y, preds)
        return score