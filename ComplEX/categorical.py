import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from stellargraph import utils
import Categorizer
import model
import neo_graph

test_frac = 0.3
train_frac = 0.999
val_frac = 0.3
cv = 2#10
iter = 1#5
negative_samples = 10
epochs = 15

# name of the file to save the size of the test and training set
dataset_size_file = 'data/categorical_dataset_size'

# Retrieve node matrix and subgraph with chemical-causes-side effect relations.
# discrete decides if continuous frequencies or frequency classes are given back
# frequency_as_type=True puts the edge classes 0, 1, 2, 3, 4 or 5 as labels
# frequency_as_type=False puts 0 if edge doen't exit and 1 if edge exists as labels
nodes = neo_graph.get_nodes(frequency_as_type=True)
adr_graph = neo_graph.graph_from_nodes(nodes)

# plot a histogram of the frequency classes (0 to 6) in all retrieved data
nodes.rename(columns={"type": "label"}, inplace=True)
neo_graph.plot_freq_hist(nodes['label'].values, 'data/complex_hist_cat.png')

# sample a certain number of labels from each class with replacement
new_nodes = nodes.groupby('label').sample(n=10000, replace=True)
new_nodes = new_nodes.sample(frac=1)

# split edges into a training and a test set
train, test = train_test_split(new_nodes, test_size=test_frac)

# save the sizes of training and test set
with open(dataset_size_file, 'w') as f:
    f.write(f'test: {len(test.index)}')
    f.write(f'train: {len(train.index)}')

# build a categorizer to use for hyperparameter optimization
cat = Categorizer.Categorizer(epochs, negative_samples, False, adr_graph, 1e-7, 100, train)

# dictionary for hyperparameter optimization
p_dicts = [{'l2_reg': np.linspace(0, 1e-7, 10), 'embedding_dimension': [50, 100, 150, 200]}]

# hyperparameter optimization
search = RandomizedSearchCV(estimator=cat, param_distributions=p_dicts, n_iter=iter, cv=cv, error_score='raise')
search.fit(train)

best_est = search.best_estimator_
best_params = search.best_params_

# save best-found parameters from hyperparameter optimization
with open('data/hyp_cat', 'w') as f:
    f.write(str(best_params))

# make train-val splits for cross-validation of thresholds
kf = KFold(n_splits=10)
splits = []
for train_index, test_index in kf.split(train):
    t_train, t_val = train.iloc[train_index], train.iloc[test_index]
    split = (t_train, t_val)
    splits.append(split)

# search optimal thresholds on the training set
adr_gen, adr_complex, adr_model, adr_history, scores, threshs = model.train_model(False, adr_graph, splits, **best_params)

# save thresholds
with open('data/threshs', 'w') as f:
    f.write(str(threshs))

# final evaluation
model.categorical_evaluation_final(adr_graph, best_est.adr_model_, test, threshs)

# plot performance on training / validation set
utils.plot_history(best_est.adr_history_, return_figure=True)
plt.savefig('data/categorical_tr_curves')

# rank evaluation
test.sort_values(by=['label'], inplace=True)
adr_raw_ranks, adr_filtered_ranks = best_est.adr_complex.rank_edges_against_all_nodes(
    best_est.adr_gen.flow(test), adr_graph
)
# save ranking results
with open('data/categorical_results_mrr.txt', 'w') as f:
    f.write(str(model.summarise({"raw": adr_raw_ranks, "filtered": adr_filtered_ranks})))