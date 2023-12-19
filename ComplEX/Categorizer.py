from copy import deepcopy
import keras
import matplotlib as mpl
mpl.use('Qt5Agg')
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import numpy as np
from keras import callbacks, optimizers, losses, metrics, regularizers, Model
from stellargraph.layer import ComplEx
from stellargraph.mapper import KGTripleGenerator
import model


# Does categorical classification.
class Categorizer(BaseEstimator, ClassifierMixin):
    # Initializes the categorizer.
    def __init__(self, epochs, negative_samples, binary, adr_graph, l2_reg, embedding_dimension, X, thresholds=None, classes=None):
        self.negative_samples = negative_samples
        self.adr_graph = adr_graph
        self.binary = binary
        self.epochs = epochs

        # hyperparameters
        self.l2_reg = l2_reg
        self.embedding_dimension = embedding_dimension

        # dataframe with edges used for training and validation
        self.X = X

        # thresholds at which maximum likelihood estimate changes and corresponding class estimates
        self.thresholds = thresholds
        self.classes = classes


    # Fits the categorizer to the training data.
    def fit(self, train, y=None):
        scores = []
        thresholds = []

        # perform split into training and test set
        adr_train, adr_valid = train_test_split(train, test_size=0.3)

        # print fraction of 0s / 1s in splits for data set analysis
        print('fit')
        print(f'training set length: {len(adr_train)}')

        # For binary edges set the label of positive edges to 'CAUSES_CHcSE', ie. their label in the graph,
        # so they can be processed in ComplEx model. Also strip the negative edges from validation set. ComplEx can't
        # make predictions for them, because they are not in the graph and therefore have no embeddings.
        # adr_valid_whole contains positive and negative edges and is saved for evaluation.
        if self.binary:
            adr_train = adr_train[adr_train['label'] == 1]
            adr_train['label'] = 'CAUSES_CHcSE'
            adr_train.reset_index(inplace=True, drop=True)

            adr_valid_whole = deepcopy(adr_valid)
            adr_valid = adr_valid[adr_valid['label'] == 1]
            adr_valid['label'] = 'CAUSES_CHcSE'
            adr_valid.reset_index(inplace=True, drop=True)

            print(f'positive training set length: {len(adr_train)}')
            print(f'validation set length: {len(adr_valid_whole)}')
            print(f'positive training set length: {len(adr_valid)}')

        # make edge generator for graph edges
        self.adr_gen = KGTripleGenerator(
            self.adr_graph, batch_size=len(adr_train) // 100
        )

        # construct the embeddings
        self.adr_complex = ComplEx(
            self.adr_gen,
            embedding_dimension=self.embedding_dimension,
            embeddings_regularizer=regularizers.l2(self.l2_reg),  # 1e-7
        )
        adr_inp, adr_out = self.adr_complex.in_out_tensors()

        # initialize the model
        self.adr_model_ = Model(inputs=adr_inp, outputs=adr_out)
        self.adr_model_.compile(
            optimizer=optimizers.Adam(lr=0.0005),
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.0)],

        )

        # make edge generator for training edges
        adr_train_gen = self.adr_gen.flow(
            adr_train, negative_samples=self.negative_samples, shuffle=True
        )

        # make edge generator for validation edges
        adr_valid_gen = self.adr_gen.flow(adr_valid, negative_samples=self.negative_samples)

        adr_es = callbacks.EarlyStopping(monitor="val_loss", patience=10)

        # fit the model
        self.adr_history_ = self.adr_model_.fit(
            adr_train_gen, validation_data=adr_valid_gen, epochs=self.epochs, callbacks=[adr_es]
        )
        # evaluate model on validation set based the type of data (binary / categorical) that is used
        if self.binary:
            score, thresh = model.binary_evaluation(self.adr_graph, self.adr_model_, adr_valid_whole)
            scores.append(score)
            thresholds.append(thresh)
        else:
            max_scorers_per_class, max_threshs_per_class = model.categorical_evaluation(self.adr_graph, self.adr_model_, adr_valid)
            scores.append(max_scorers_per_class)
            thresholds.append(max_threshs_per_class)

        return self

    # Scores the predictions for edges in dataframe X made by the fitted classifier against their true labels.
    def score(self, X, y=None, perClass = False):
        self.max_scorers_per_class_, self.max_threshs_per_class_ = model.categorical_evaluation(self.adr_graph, self.adr_model_, X)
        mean_f1 = np.mean(self.max_scorers_per_class_[0])

        return mean_f1