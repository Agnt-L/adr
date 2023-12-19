import sys
import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Integer
import Classifier
from neuralfingerprint import binary_classification_nll
matplotlib.use('Qt5Agg')
import neo_graph
import os
import autograd.numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
import matplotlib.pyplot as plt
from neuralfingerprint import load_data, relu
from neuralfingerprint import build_conv_deep_net, build_convnet_fingerprint_fun
from neuralfingerprint import degrees, build_standard_net
from neuralfingerprint.data_util import remove_duplicates
import evaluation
from neuralfingerprint.build_vanilla_net import sigmoid
import pandas as pd

log = sys.stderr.write


# parameters used for training the neural network
batch_size = 10
train_frac = 0.7
test_frac = 0.3

# parameters for running the algorithm
params = dict(num_records = 20,
              model = dict(net_type = 'conv_plus_net',   # 'morgan' | 'conv'
                           fp_length = 50,   #
                           fp_depth = 3,   #
                           conv_width = 10, #      # conv net only
                           h1_size = 100,   #
                           L2_reg = np.exp(-6),),   #
              train = dict(num_iters = 100,
                           batch_size = batch_size,
                           init_scale = np.exp(-4),  #
                           step_size = np.exp(-1),  #
                           seed = 0,),
              name = 'asep_nec',
              b1 = np.exp(-4),
              b2 = np.exp(-4),
              l1_penalty = np.exp(-5))

# information about the datasets to be used
datasets_info = dict(
    derm = dict(
            nll_func = binary_classification_nll,
            target_name = 'freq',
            data_file ='dermatitis_perioral'),
    back = dict(
                nll_func = binary_classification_nll,
                target_name = 'freq',
                data_file ='back_pain'),
    asep_nec = dict(
                    nll_func = binary_classification_nll,
                    target_name = 'freq',
                    data_file ='aseptic_necrosis'))

# saves best found hyperparameters
best_params = {'fp_length': 50,
            'fp_depth': 3,
            'init_scale':np.exp(-4),
            'step_size':np.exp(-5),
                    'b1':np.exp(-4),
                    'b2':np.exp(-4),
            'L2_reg':np.exp(-6),
            'l1_penalty':np.exp(-5),
            'conv_width':10}



# plotting parameters
num_figs_per_fp = 1
figsize = (100, 100)
highlight_color = (30.0/255.0, 100.0/255.0, 255.0/255.0)  # A nice light blue.


# Draws given molecule and highlight given highlight_atoms.
def draw_molecule_with_highlights(filename, smiles, highlight_atoms):
    drawoptions = DrawingOptions()
    drawoptions.selectColor = highlight_color
    drawoptions.elemDict = {}   # Don't color nodes based on their element.
    drawoptions.bgColor=None

    mol = Chem.MolFromSmiles(smiles)
    fig = Draw.MolToMPL(mol, highlightAtoms=highlight_atoms, size=figsize, options=drawoptions,fitImage=False)

    plt.title(str(smiles))

    fig.gca().set_axis_off()
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# parameters used to backtrack activation of the atoms
num_epochs = 100
normalize = 1
dropout = 0
activation = relu

# Parses given parameters to be used to recreate trained network.
def parse_training_params(params):
    nn_train_params = {'num_epochs'  : num_epochs,
                       'batch_size'  : batch_size,
                       'learn_rate'  : params['step_size'],
                       'b1'          : params['b1'],
                       'b2'          : params['b2'],
                       'param_scale' : params['init_scale']}

    vanilla_net_params = {'layer_sizes':[params['fp_length']],  # Linear regression.
                          'normalize':normalize,
                          'L2_reg': params['L2_reg'],
                          'L1_reg': params['l1_penalty'],
                          'activation_function':activation}
    return nn_train_params, vanilla_net_params

# Constructs array of all the neighbor atoms of the given atoms.
def construct_atom_neighbor_list(array_rep):
    atom_neighbour_list = []
    for degree in degrees:
        atom_neighbour_list += [list(neighbours) for neighbours in array_rep[('atom_neighbors', degree)]]
    return atom_neighbour_list

# Produces pictures of num_figs_per_fp most predictive substructures (over all radii) for each fingerprint index
# highlighted in the corresponding molecule.
def plot(trained_weights):
    # get data
    dataset_info = datasets_info[params['name']]
    data_dir = os.path.join(os.path.dirname(__file__), 'data/')
    full_data_path = os.path.join(data_dir, dataset_info['data_file'])

    print("Loading training data...")


    data = pd.read_csv(full_data_path + '.csv', usecols = ['smiles', dataset_info['target_name']])
    X = data['smiles'].values
    y = data['freq'].values
    train_smiles, train_targets, _, _ = train_test_split(X, y, test_size=int(np.floor(test_frac*len(X) / batch_size) * batch_size),
                                                        train_size=int(np.floor(train_frac*len(X) / batch_size) * batch_size))

    # make fingerprint network
    print("Convnet fingerprints with neural net")
    conv_arch_params['return_atom_activations'] = True
    output_layer_fun, parser, compute_atom_activations = \
       build_convnet_fingerprint_fun(**conv_arch_params)
    # get atom activations from last layer and array representation of molecule graph
    atom_activations, array_rep = compute_atom_activations(trained_weights, train_smiles)

    # make directory for figures
    if not os.path.exists('figures'): os.makedirs('figures')

    # make dictionary of parent molecules for the atoms
    parent_molecule_dict = {}
    for mol_ix, atom_ixs in enumerate(array_rep['atom_list']):
        for atom_ix in atom_ixs:
            parent_molecule_dict[atom_ix] = mol_ix

    atom_neighbor_list = construct_atom_neighbor_list(array_rep)

    # Recursive function to get indices of all atoms in a certain radius.
    def get_neighborhood_ixs(array_rep, cur_atom_ix, radius):
        if radius == 0:
            return set([cur_atom_ix])
        else:
            cur_set = set([cur_atom_ix])
            for n_ix in atom_neighbor_list[cur_atom_ix]:
                cur_set.update(get_neighborhood_ixs(array_rep, n_ix, radius-1))
            return cur_set

    # Recreate trained network.
    nn_train_params, vanilla_net_params = parse_training_params(best_params)
    conv_arch_params['return_atom_activations'] = False
    _, _, combined_parser = \
        build_conv_deep_net(conv_arch_params, vanilla_net_params, best_params['L2_reg'])

    net_loss_fun, net_pred_fun, net_parser = build_standard_net(**vanilla_net_params)
    net_weights = combined_parser.get(trained_weights, 'net weights')
    last_layer_weights = net_parser.get(net_weights, ('weights', 0))

    # iterate over all fingerprint indices
    for fp_ix in range(best_params['fp_length']):
        print("FP {0} has linear regression coefficient {1}".format(fp_ix, last_layer_weights[fp_ix][0]))

        # make list of index of the atom 'atom_ix', the radius of the corresponding substructure 'radius' and how much
        # this substructure activates the current fingerprint index 'fp_activation'
        combined_list = []
        for radius in all_radii:
            fp_activations = atom_activations[radius][:, fp_ix]
            combined_list += [(fp_activation, atom_ix, radius) for atom_ix, fp_activation in enumerate(fp_activations)]

        unique_list = remove_duplicates(combined_list, key_lambda=lambda x: x[0])
        combined_list = sorted(unique_list, key=lambda x: -x[0])

        # iterate over a certain number of most activating substructures to draw
        for fig_ix in range(num_figs_per_fp):
            # expand fig_ix most activating substructure to get list of atoms it contains
            activation, most_active_atom_ix, cur_radius = combined_list[fig_ix]
            most_activating_mol_ix = parent_molecule_dict[most_active_atom_ix]
            highlight_list_our_ixs = get_neighborhood_ixs(array_rep, most_active_atom_ix, cur_radius)
            highlight_list_rdkit = [array_rep['rdkit_ix'][our_ix] for our_ix in highlight_list_our_ixs]

            print("radius:", cur_radius, "atom list:", highlight_list_rdkit, "activation", activation)

            # plot the molecule with the fig_ix most activating substructure highlighted
            draw_molecule_with_highlights(
                "figures/fp_{0}_highlight_{1}.pdf".format(fp_ix, fig_ix),
                train_smiles[most_activating_mol_ix],
                highlight_atoms=highlight_list_rdkit)
    # weights of the last layer are the regression coefficients of linear regression
    return last_layer_weights

# Reads csv files and returns input data, positive examples, negative examples and the nll function.
# The smiles strings have to be in a column named 'smiles'.
# The target column must have the specified 'target_name'.
def load_task_data(name):
    dataset_info = datasets_info[name]
    data_dir = os.path.join(os.path.dirname(__file__), 'data/')
    full_data_path = os.path.join(data_dir, dataset_info['data_file'])


    data = pd.read_csv(full_data_path + '.csv', usecols = ['smiles', dataset_info['target_name']])
    data_pos = pd.read_csv(full_data_path + '_pos.csv', usecols=['smiles', dataset_info['target_name']])
    data_neg = pd.read_csv(full_data_path + '_neg.csv', usecols=['smiles', dataset_info['target_name']])

    return data, data_pos, data_neg, dataset_info['nll_func']

# Produces n_splits equal sized folds of indices within data_size.
def custom_KFold(data_size, n_splits):
    folds = []

    fold_size = int(np.floor((data_size / n_splits) / batch_size) * batch_size)

    for i in range(0, n_splits):
        indices_val = list(range(i * fold_size, (i+1) * fold_size))
        indices_train = list(range(0, i * fold_size)) + list(range((i+1) * fold_size, n_splits * fold_size))
        folds.append((indices_train, indices_val))

    return folds

# Produces the input data sets for the experiments of the thesis.
def make_data_sets():
    neo_graph.assocs_per_se('Dermatitis perioral', 100, 'data/dermatitis_perioral')
    neo_graph.assocs_per_se('Back pain', 5000, 'data/back_pain')
    neo_graph.assocs_per_se('Aseptic necrosis', 100, 'data/aseptic_necrosis')

def main():
    #make_data_sets()

    # load input data and perform train-test splits for positive and negative examples individually to have an equal
    # amount of positive and negative examples in the training / test set
    data, data_pos, data_neg, nll_func = load_task_data(name=params['name'])
    X = data['smiles'].values
    y = data['freq'].values
    pos_train, pos_test = train_test_split(data_pos, test_size=int(np.floor(test_frac*len(data_pos) / batch_size) * batch_size),
                                                        train_size=int(np.floor(train_frac*len(data_pos) / batch_size) * batch_size))
    neg_train, neg_test = train_test_split(data_neg, test_size=int(np.floor(test_frac*len(data_neg) / batch_size) * batch_size),
                                                        train_size=int(np.floor(train_frac*len(data_neg) / batch_size) * batch_size))

    # make a histogram for the input labels
    neo_graph.plot_freq_hist(y, 'data/dl_hist')

    # concatenate positive and negative examples to form the whole train / test set
    train = pd.concat([pos_train, neg_train])
    train.sample(frac=1)
    test = pd.concat([pos_test, neg_test])
    test.sample(frac=1)

    # seperate into the feature and target values
    X_train = train['smiles'].values
    X_test = test['smiles'].values
    y_train = train['freq'].values
    y_test = test['freq'].values

    # make histogram of the labels of the test set
    neo_graph.plot_freq_hist(y_test, 'data/dl_hist_test')

    # search for a particular substructure among the drug molecules
    # not necessary to run the algorithm
    """pattern = Chem.MolFromSmarts('[#6](-[#6])(=[#7])-[#7]')

    for idx, smiles in enumerate(X):
        m = Chem.MolFromSmiles(smiles)
        causesADR = y[idx]
        print("Structure {}: pattern found {}, causes adr: {}".format(idx, m.HasSubstructMatch(pattern), causesADR))"""

    # hyperparameters to be explored by genetic algorithm
    param_grid = {
        'fp_length': Integer(21, 100), # model params
        'fp_depth': Integer(1, 4),
        'conv_width': Integer(5, 20),
        'h1_size': Integer(50, 200),
        'L2_reg': Continuous(np.exp(-6), np.exp(-4)),
        'init_scale': Continuous(np.exp(-5), np.exp(-3)),  # train params
        'step_size': Continuous(np.exp(-8), np.exp(-1))
    }

    # make custom cv-split with slices to be used by genetic algorithm
    cv = custom_KFold(len(X_train), n_splits=3)

    # fit a simple classifier with given hyperparameters
    pm = params['model']
    pt = params['train']
    clf = Classifier.Classifier(params, pm['fp_length'], pm['fp_depth'], pm['conv_width'], pm['h1_size'], pm['L2_reg'], pt['init_scale'], pt['step_size'], nll_func)
    clf.fit(X_train, y_train)

    # use genetic algorithm to find good hyperparameters
    evolved_estimator = GASearchCV(estimator=clf,
                                   cv=cv,
                                   population_size=3,
                                   generations=2,
                                   tournament_size=2,
                                   elitism=True,
                                   crossover_probability=0.8,
                                   mutation_probability=0.1,
                                   param_grid=param_grid,
                                   criteria='max',
                                   algorithm='eaMuPlusLambda',
                                   n_jobs=-1,
                                   verbose=True,
                                   keep_top_k=1)
    evolved_estimator.fit(X_train, y_train)
    score = evolved_estimator.score(X_test, y_test)
    print(score)
    print(evolved_estimator.best_params_)

    # update best_params dict with best parameters found by genetic algorithm
    best_params.update(evolved_estimator.best_params_)

    # define global hyperparameters needed to backtrack atom activations
    global conv_layer_sizes
    global conv_arch_params
    global all_radii

    # calculate them based on best found hyperparameters
    conv_layer_sizes = [best_params['conv_width']] * best_params['fp_depth']
    conv_arch_params = {'num_hidden_features': conv_layer_sizes,
                        'fp_length': best_params['fp_length'],
                        'normalize': normalize,
                        'return_atom_activations': False}
    all_radii = list(range(best_params['fp_depth'] + 1))


    # get most predictive fingerprint index
    regression_coefficients = plot(evolved_estimator.best_estimator_.weights_)
    most_pred_fingerprint = np.argmax([abs(x[0]) for x in regression_coefficients.tolist()])


    # evaluation of classfier using best hyperparameters found by genetic algorithm
    y_pred = evolved_estimator.predict(X_test)
    sig_pred = [sigmoid(x) for x in y_pred]

    best_est = evolved_estimator.best_estimator_
    best_est.plot_train_val_curves('data/train_val_curves_' + params['name'])

    evaluation.binary_evaluation_final(y_test.tolist(), sig_pred, 'dl_eval_' + params['name'],
                                                                                           'Deep Learning', [0.5, 0.5, 0.5])

    with open('data/dl_eval_' + params['name'], 'a') as f:
        f.write(str(evolved_estimator.best_params_))
main()