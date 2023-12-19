import matplotlib.pyplot as plt
from stellargraph import utils
import numpy as np
import model
import neo_graph

test_frac = 0.3
train_frac = 0.999
val_frac = 0.3
cv = 10
iter = 5

# name of the file to save the size of the test and training set
dataset_size_file = 'data/binary_dataset_size'

# Retrieve node matrix and subgraph with chemical-causes-side effect relations.
# discrete decides if continuous frequencies or frequency classes are given back
# frequency_as_type=True puts the edge classes 0, 1, 2, 3, 4 or 5 as labels
# frequency_as_type=False puts 0 if edge doen't exit and 1 if edge exists as labels
nodes = neo_graph.get_nodes(frequency_as_type=False)
adr_graph = neo_graph.graph_from_nodes(nodes)

# plot a histogram of the frequency classes (0 and 1) in all retrieved data
neo_graph.plot_freq_hist(nodes['type'].values, 'data/complex_hist_bin.png')
print("test_set")

# sample test edges and write size of the test size into dataset_size_file
with open(dataset_size_file, 'w') as f:
    f.write('test: ')
test_df, graph_without_test = model.get_test_edges(adr_graph, True, test_frac, dataset_size_file)

# plot a histogram of the frequency clases (0 or 1) in the test set
neo_graph.plot_freq_hist(test_df['label'].values, 'data/complex_hist_test_bin.png')

# make splits for cross validation
print('\n')
print('cv')
cv_splits = model.train_val_splitCV(adr_graph, cv, graph_without_test, train_frac, val_frac, True,
                                    dataset_size_file)

# dictionary for hyperparameter optimization
p_dicts = [{'l2_reg': np.linspace(0, 1e-7, 10)}, {'embedding_dimension': [50, 100, 150, 200]}]

# get dict of best hyperparameters over cv_splits
td, thresh = model.randomSearchCV(adr_graph, p_dicts, iter, cv, train_frac=train_frac, val_frac=val_frac, binary=True,
                          data_type='bin')

with open('data/hyp_bin', 'w') as f:
    f.write(str(td))

# sample test edges and write size of the test size into dataset_size_file
# (use almost all edges from graph without the test edges as final training edges,
# fraction of 1 is not allowed in edge sampler)
print("training_set")
with open(dataset_size_file, 'a') as f:
    f.write('train: ')
final_train_df, graph_without_final_train = model.get_test_edges(graph_without_test, True, 0.99,
                                                                 dataset_size_file)

# list of final data split
final_cv_splits = [(final_train_df, test_df)]

# train final model with best hyperparameters
adr_gen, adr_complex, adr_model, adr_history, scores, _ = model.train_model(True, adr_graph, final_cv_splits, **td)

# do final evaluation and save results in data folder
model.binary_evaluation_final(adr_graph, adr_model, test_df, thresh)

# save thresholds
with open('data/thresh', 'w') as f:
    f.write(str(thresh))

# save training and validation curve
utils.plot_history(adr_history, return_figure=True)
plt.savefig('data/complex_tr_curves')

# filter the test set to contain only the positive examples use ranking evaluation metrics (not used in thesis)
test_df = test_df[test_df['label'] == 1]
test_df['label'] = 'CAUSES_CHcSE'
test_df.reset_index(inplace=True, drop=True)

# rank scores of true edges against predicted edges
# The rank of the score of an edge in between all edges were the subject is changed and the ones were the object
# is changed is calculated.
# For adr_filtered_ranks the edges that are in present in the graph are ignored in the ranking.
adr_raw_ranks, adr_filtered_ranks = adr_complex.rank_edges_against_all_nodes(
    adr_gen.flow(test_df), adr_graph
)

# save ranking results
with open('data/ComplEx_binary_results_mrr.txt', 'w') as f:
    f.write(str(model.summarise({"raw": adr_raw_ranks, "filtered": adr_filtered_ranks})))