import random
from copy import deepcopy

import keras
import matplotlib as mpl

mpl.use('Qt5Agg')
import numpy as np
import pandas as pd
from keras import callbacks, optimizers, losses, metrics, regularizers, Model
from sklearn.metrics import matthews_corrcoef, precision_recall_curve
from sklearn.metrics import f1_score, accuracy_score, auc
from stellargraph.data import EdgeSplitter
from stellargraph.layer import ComplEx
from stellargraph.mapper import KGTripleGenerator
import model
import evaluation

epochs = 15
negative_samples = 10

# Samples parameters of p_dict n_iter times to get the parameters for hyperparameter optimization iterations.
def randomParameterSampler(p_dicts, n_iter):
    trial_dicts = []
    for i in range(n_iter):
        new_dict = {}
        for p_dict in p_dicts:
            for var in p_dict:
                p_list = p_dict[var]
                rand_val = random.choice(p_list)
                new_dict[var] = rand_val
        trial_dicts.append(new_dict)
    return trial_dicts

# Custom random search similar to RandomSearchCV of sklearn but it samples an equal fraction of positive and negative
# edges directly from the graph for each model. It is used only for the binary implementation.
# Uses randomParameterSampler to obtain parameter combinations to try out.
def randomSearchCV(adr_graph, p_dicts, n_iter, cv, train_frac, val_frac, binary, data_type):
    trial_dicts = randomParameterSampler(p_dicts, n_iter)
    trial_scores = []
    thresh_arr = []
    # For each sampled hyperparameter combination:
    for td in trial_dicts:
        # Train cv (cv = number of cross-validation splits) models
        # and save scores (F1-score, Accuracy and Matthews Correlation Coefficient) for each trained model.
        test_df, graph_without_test = model.get_test_edges(adr_graph, binary, 0.3)
        cv_splits = model.train_val_splitCV(adr_graph, cv, graph_without_test, train_frac, val_frac, binary)
        adr_gen, adr_complex, adr_model, adr_history, scores, threshs = train_model(binary, adr_graph, cv_splits, **td)

        # Save mean thresholds (for f1, acc and mcc)
        # over the cross-validation models for current hyperparamter combination td.
        thresh_arr.append(threshs)

        # Take the mean of all the scores of the cross-validation models for current hyperparamter combination td
        # and save it.
        cv_mean_score = np.mean(scores)
        trial_scores.append(cv_mean_score)

    # Find the maximum cross-validated score of models trained with different hyperparameters.
    max_idx = np.argmax(trial_scores)
    print(f'trial_scores: {trial_scores}')
    print(f'trial_dicts: {trial_dicts}')
    # Return hyperparameters and  for best cross-validated model
    return trial_dicts[max_idx], thresh_arr[max_idx]

# Trains the model with fitting metric and loss function based on given data type binary = True
# or binary = False (categorical).
# regression 'reg'. Returns adr_gen, adr_complex, adr_model, adr_history, scores and mean_threshs.
# adr_gen: an edge generator used for turning a dataframe of edges into edge representation
#          necessary for the ranking method.
# adr_complex: saves the complex embeddings
# adr_model: contains the trained model
# adr_history: saves mrr and best k hits of the model
# scores: saves f1, acc and mcc for all trained models
# mean_threshs: saves mean thresholds, each over models optimized for f1, acc or mcc respectively
def train_model(binary, adr_graph, cv_splits, l2_reg, embedding_dimension):
    scores = []
    thresholds = []

    for cv in cv_splits:
        adr_train = cv[0]
        adr_valid = cv[1]

        # print fraction of 0s / 1s in splits for data set analysis
        print('fit')
        print(f'training set length: {len(adr_train)}')

        # For binary edges set the label of positive edges to 'CAUSES_CHcSE', ie. their label in the graph,
        # so they can be processed in ComplEx model. Also strip the negative edges from validation set. ComplEx can't
        # make predictions for them, because they are not in the graph and therefore have no embeddings.
        # adr_valid_whole contains positive and negative edges and is saved for evaluation.
        if binary:
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
        adr_gen = KGTripleGenerator(
            adr_graph, batch_size=len(adr_train) // 100
        )

        # construct the embeddings
        adr_complex = ComplEx(
            adr_gen,
            embedding_dimension=embedding_dimension,
            embeddings_regularizer=regularizers.l2(l2_reg), #1e-7
        )
        adr_inp, adr_out = adr_complex.in_out_tensors()

        # initialize the model
        adr_model = Model(inputs=adr_inp, outputs=adr_out)
        adr_model.compile(
            optimizer=optimizers.Adam(lr=0.0005),
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.0)],

        )

        # make edge generator for training edges
        adr_train_gen = adr_gen.flow(
            adr_train, negative_samples=negative_samples, shuffle=True
        )

        # make edge generator for validation edges
        adr_valid_gen = adr_gen.flow(adr_valid, negative_samples=negative_samples)

        adr_es = callbacks.EarlyStopping(monitor="val_loss", patience=10)

        # fit the model
        adr_history = adr_model.fit(
            adr_train_gen, validation_data=adr_valid_gen, epochs=epochs, callbacks=[adr_es]
        )
        # evaluate model on validation set based the type of data (binary / categorical) that is used
        if binary:
            score, thresh = binary_evaluation(adr_graph, adr_model, adr_valid_whole)
            scores.append(score[0])
            thresholds.append(thresh)
        else:
            max_scorers_per_class, max_threshs_per_class = categorical_evaluation(adr_graph, adr_model, adr_valid)
            scores.append(max_scorers_per_class)
            thresholds.append(max_threshs_per_class)
        mean_threshs = np.mean(np.array(thresholds), axis=0)
    return adr_gen, adr_complex, adr_model, adr_history, scores, mean_threshs


# Edge sampling for test edges.
# Randomly sample a fraction frac of all positive links, and same number of negative links, from graph, and obtain the
# reduced graph graph_without_test with the sampled links examples_test removed.
# Negative links are pairs which don't have the given edge type as a relation.
# In the categorical case non-zero edge labels need to be looked up in the graph which is done by method
# labels_categorical.
def get_test_edges(adr_graph, binary, frac, eval_file=None):
    # Define an edge splitter on the original graph:
    edge_splitter_test = EdgeSplitter(adr_graph)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    graph_without_test, examples_test, labels_test = edge_splitter_test.train_test_split(
        p=frac, method="global"
    )
    if eval_file is not None:
        with open(eval_file, 'a') as f:
            f.write(f'positives: {len(edge_splitter_test.positive_edges_ids)}, '
                    f'negatives: {len(edge_splitter_test.negative_edges_ids)}')
            f.write('\n')

    examples_list = examples_test.tolist()
    labels_list = labels_test.tolist()
    test_list = [examples_list[i] + [labels_list[i]] for i in range(len(examples_test))]
    test_df = pd.DataFrame(test_list, columns=['source', 'target', 'label'])

    if not binary:
        edge_classes, true_edge_classes = labels_categorical(adr_graph, test_df)
        test_df['labels'] = edge_classes.values.tolist()

    return test_df, graph_without_test

# Uses same splitting strategy as get_test_edges(..) but samples a validation and a training set and does that
# cv number of times.
# Returns array with cv (train, val) tuples. Result can be used as a splitting strategy with sklearn methods.
def train_val_splitCV(graph, cv, graph_without_test, train_frac, val_frac, binary, eval_file=None):
    edge_splitter_valid = EdgeSplitter(graph_without_test, graph)

    cv_splits = []
    for i in range(cv):
        print('val')
        graph_without_val, examples_valid, labels_valid = edge_splitter_valid.train_test_split(
            p=val_frac, method="global"
        )
        if eval_file is not None:
            with open(eval_file, 'a') as f:
                f.write('valid\n')
                f.write(f'positives: {len(edge_splitter_valid.positive_edges_ids)}, '
                        f'negatives: {len(edge_splitter_valid.negative_edges_ids)}')
                f.write('\n')


        examples_list = examples_valid.tolist()
        labels_list = labels_valid.tolist()
        valid_list = [examples_list[i] + [labels_list[i]] for i in range(len(examples_valid))]
        valid_df = pd.DataFrame(valid_list, columns=['source', 'target', 'label'])

        if not binary:
            edge_classes, true_edge_classes = labels_categorical(graph, valid_df)
            valid_df['labels'] = edge_classes.values.tolist()

        print('train')
        edge_splitter_train = EdgeSplitter(graph_without_val, graph)
        adr_tr, examples_train, labels_train = edge_splitter_train.train_test_split(
            p=train_frac, method="global"
        )
        if eval_file is not None:
            with open(eval_file, 'a') as f:
                f.write('train\n')
                f.write(f'positives: {len(edge_splitter_train.positive_edges_ids)}, '
                        f'negatives: {len(edge_splitter_train.negative_edges_ids)}')
                f.write('\n')

        examples_list = examples_train.tolist()
        labels_list = labels_train.tolist()
        train_list = [examples_list[i] + [labels_list[i]] for i in range(len(examples_train))]
        train_df = pd.DataFrame(train_list, columns=['source', 'target', 'label'])

        if not binary:
            edge_classes, true_edge_classes = labels_categorical(graph, train_df)
            train_df['labels'] = edge_classes.values.tolist()


        cv_splits.append((train_df, valid_df))
    return cv_splits


# Helper function to return ranking metrics as a dataframe.
def results_as_dataframe(name_to_results):
    return pd.DataFrame(
        name_to_results.values(),
        columns=["mrr", "hits at 1", "hits at 3", "hits at 10"],
        index=name_to_results.keys(),
    )

# Calculates MRR and instances of edges being under the first 1, 3 or 10 in the ranking.
def summarise(name_to_ranks):
    return results_as_dataframe(
        {
            name: (
                np.mean(1 / ranks),
                np.mean(ranks <= 1),
                np.mean(ranks < 3),
                np.mean(ranks <= 10),
            )
            for name, ranks in name_to_ranks.items()
        }
    )

# Finds the relation labels from adr_valid dataframe in adr_graph and returns their frequency values, ie. edge types.
# Negative edges have just 0 as their frequency like their labels.
def labels_categorical(adr_graph, adr_valid):
    examples_test = adr_valid[['source', 'target']].values.tolist()
    labels_test = adr_valid['label'].values.tolist()
    graph_edges = pd.DataFrame(adr_graph.edge_arrays(True)[0:3]).transpose()
    pos_zero = labels_test.index(0)
    true_ex = examples_test[0:pos_zero]
    false_ex = pd.DataFrame(examples_test[pos_zero:len(examples_test)])
    true_labels = [graph_edges[2][(graph_edges[0] == ex[0]) & (graph_edges[1] == ex[1])] for ex in true_ex]
    true_labels = pd.DataFrame([x.values for x in true_labels], columns=["label"])
    false_labels = pd.DataFrame(labels_test[pos_zero:len(examples_test)], columns=["label"])

    labels = pd.concat([true_labels, false_labels])
    return labels, true_labels

# Generates rows with each possible frequency type for each positive edge in adr_valid to be scored by ComplEx.
def examples_to_score(adr_valid):
    examples_test = adr_valid[['source', 'target']].values.tolist()
    labels_test = adr_valid['label'].values.tolist()

    pos_zero = labels_test.index(0)
    true_ex = examples_test[0:pos_zero]

    examples_all_labels = [e + [i] for e in examples_test for i in range(0, 6)]
    examples_all_labels = pd.DataFrame(examples_all_labels, columns=["source", "target", "label"])
    return examples_all_labels


# Does categorical evaluation in the training process. Returns the best thresholds for each metric per class and
# the best metric values resulting from them.
def categorical_evaluation(adr_graph, adr_model, valid):
    # get example edges
    examples = examples_to_score(valid)

    true_labels = valid[['label']]

    # make predictions for the example edges
    pred_edges_gen = KGTripleGenerator(
        adr_graph, batch_size=len(examples) // 100
    )
    pred_sequence = pred_edges_gen.flow(examples)
    predic = pd.DataFrame(adr_model.predict(pred_sequence), columns=["pred"])

    # array for saving evaluation curves
    pr_aucs_per_class = []
    pr_auc_curves_per_class = []

    # metrics to calculate
    scorers = {
        'f1_score': f1_score,
        'acc': accuracy_score,
        'mcc': matthews_corrcoef
    }
    max_scorers_per_class = []
    max_threshs_per_class = []

    for i in range(len(scorers)):
        max_scorers_per_class.append([])
        max_threshs_per_class.append([])

    # iterate over freqeuncy classes
    for c in range(0, 6):
        # get indices of all the example edges for class label c
        idx = examples.index[examples['label'] == c].tolist()
        # get predicted scores at these indices
        score = [x for x in predic.iloc[idx]["pred"]]
        # make an array signaling which labels are really of class c
        y_true = [x[0] == c for x in true_labels.values.tolist()]

        # calculate pr auc curve and pr auc for class c
        precision, recall, thresholds = precision_recall_curve(y_true, score)
        pr_auc_curves_per_class.append((precision, recall, thresholds))
        pr_aucs_per_class.append(auc(recall, precision))

        scorers_one_class = []
        for j in range(len(scorers)):
            scorers_one_class.append([])

        # try out a lot of thresholds in between min and max predicted score
        score_range = np.linspace(min(score), max(score), 100)
        for i in score_range:
            y_pred = [x > i for x in score]
            # calculate each metric for class c and append it to subarray j of scorers_one_class
            j = 0
            for scorer in scorers:
                scorers_one_class[j].append(scorers[scorer](y_true, y_pred))
                j += 1

        # calculate max value for each metric and corresponding threshold for class c
        j = 0
        for scorer_one_class in scorers_one_class:
            max_val = max(scorer_one_class)
            max_scorers_per_class[j].append(max_val)
            thresh = score_range[np.argmax(scorer_one_class)]
            max_threshs_per_class[j].append(thresh)
            j += 1

    # return max scores for each metric per class and corresponding best thresholds
    return max_scorers_per_class, max_threshs_per_class


# Final evaluation for categorical model. Saves evaluation results for f1-score, accuracy, mcc, pr-auc-score
# and confusion matrix.
def categorical_evaluation_final(adr_graph, adr_model, valid, threshs):
    # generate example edges with each possible frequency class
    examples = examples_to_score(valid)

    true_labels = valid[['label']]

    # predict scores for the example edges
    pred_edges_gen = KGTripleGenerator(
        adr_graph, batch_size=len(examples) // 100
    )
    pred_sequence = pred_edges_gen.flow(examples)
    predic = pd.DataFrame(adr_model.predict(pred_sequence), columns=["pred"])

    # for each frequency class
    for c in range(0, 6):
        # get indices of all the example edges for class label c
        idx = examples.index[examples['label'] == c].tolist()
        # get predicted scores at these indices
        score = [x for x in predic.iloc[idx]["pred"]]
        # make an array signaling which labels are really of class c
        y_true = [x[0] == c for x in true_labels.values.tolist()]
        # do binary evaluation for class c
        evaluation.binary_evaluation_final(y_true, score, f'categorical_results_{c}', 'ComplEx', [x[c] for x in threshs])


# Does binary evaluation in the training process.
# Returns maximal F1-score, ACC and MCC and the corresponding thresholds.
def binary_evaluation(adr_graph, adr_model, valid):
    # format validation edges and predict scores
    examples = deepcopy(valid)
    examples['label'] = 'CAUSES_CHcSE'
    pred_edges_gen = KGTripleGenerator(
        adr_graph, batch_size=len(examples) // 100  # ~10 batches per epoch
    )
    pred_sequence = pred_edges_gen.flow(examples)
    score = [x[0] for x in adr_model.predict(pred_sequence)]

    # metrics to calculate
    scorers = {
        'f1_score': f1_score,
        'acc': accuracy_score,
        'mcc': matthews_corrcoef
    }

    # get true values
    y_true = valid['label'].values.tolist()

    # make array of f1-scores using different thresholds
    max_scores = []
    best_threshs = []
    # thresholds to try between min and max score
    score_range = np.linspace(min(score), max(score), 100)

    # iterate over metrics
    for scorer in scorers:
        scores = []
        # iterate over thresholds
        for i in score_range:
            y_pred = [x > i for x in score]
            # save the score
            scores.append(scorers[scorer](y_true, y_pred))
        # get best score and corresponding threshold
        max_idx = np.argmax(scores)
        max_score = score[max_idx]
        best_thresh = score_range[max_idx]
        # save them
        max_scores.append(max_score)
        best_threshs.append(best_thresh)

    # return maximal scores and thresholds
    return max_scores, best_threshs

# Final evaluation for binary model. Predicts scores for validation edges and puts true values and scores
# into function evaluation.binary_evaluation_final.
def binary_evaluation_final(adr_graph, adr_model, valid, thresh):
    # format validation edges and predict their scores
    examples = deepcopy(valid)
    examples['label'] = 'CAUSES_CHcSE'
    pred_edges_gen = KGTripleGenerator(
        adr_graph, batch_size=len(examples) // 100  # ~10 batches per epoch
    )
    pred_sequence = pred_edges_gen.flow(examples)
    score = adr_model.predict(pred_sequence)

    # get true values
    y_true = valid['label'].values.tolist()

    # binary evaluation
    evaluation.binary_evaluation_final(y_true, score, 'complex_binary_results', 'ComplEx', thresh)