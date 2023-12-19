import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py2neo
from stellargraph import StellarDiGraph

# number of chemicals and side-effects to sample
NUM_CHEMS_SES = 50 #500
# number of unassociated chemicals and side-effects pairs to sample
LIMIT_CLASS_0 = 50 #5000

# neo4j parameters
host = "localhost"
port = "7687"
user = "neo4j"
password = "!bzu,um1Pa.NEO"


# Returns sampled nodes array either having edge types in the type column (frequency_as_type = False) or having
# frequency values in the type column (frequency_as_type = True). If the latter is the case,
# frequency values are translated to discrete frequency class types from 1 to 5.
def get_nodes(frequency_as_type):
    if frequency_as_type:
        nodes = freq_nodes_from_neo4j()

        nodes['type'] = [percent(x) for x in nodes['type']]
        nodes['type'] = [freq_class(x) for x in nodes['type']]

        delete_invalid_freqs(nodes)
    else:
        nodes = type_nodes_from_neo4j()


    return nodes

# Plots given frequencies [freqs] as a histogram with given number of bins (default bins is 20) and saves is under name.
def plot_freq_hist(freqs, name):
    fig = plt.figure()
    plt.hist(freqs, bins=20)
    plt.xlabel('frequency')
    plt.ylabel('count')
    fig.savefig(name)
    plt.close(fig)


# Samples NUM_CHEMS_SES Chemicals and Side Effects and collects the CAUSES_CHcSE relationships between them.
# Samples LIMIT_CLASS_0 non-associations between Chemicals and Side Effects.
# Returns a dataframe combining both and having columns source, target and type being 0 for non-associations and
# the frequency for known associations.
def freq_nodes_from_neo4j():
    neo4j_graph = py2neo.Graph(host=host, port=port, user=user, password=password)
    nodes = neo4j_graph.run("""
        call{
        MATCH (c:Chemical) 
        with c limit """ + str(NUM_CHEMS_SES) + """
        with collect(c) as cc
        return cc
        }
        call{
        MATCH (c:SideEffect)
        with c limit """ + str(NUM_CHEMS_SES) + """
        with collect(c) as cs
        return cs
        }
        match p=(c:Chemical) -[m:CAUSES_CHcSE]- (x:SideEffect)
        WHERE c in cc and x in cs and m.avg_frequency is not null and not m.avg_frequency = 'postmarketing'
        return id(c) as source, id(x) as target, m.avg_frequency as type
        union all
        call{
        MATCH (c:Chemical) 
        with c limit """ + str(NUM_CHEMS_SES) + """
        with collect(c) as cc
        return cc
        }
        call{
        MATCH (c:SideEffect)
        with c limit """ + str(NUM_CHEMS_SES) + """
        with collect(c) as cs
        return cs
        }
        match (c:Chemical), (x:SideEffect)
        WHERE c in cc and x in cs and not (c) -[:CAUSES_CHcSE]- (x)
        return id(c) as source, id(x) as target, 0 as type limit 
        """ + str(LIMIT_CLASS_0)).to_data_frame()
    return nodes

# Samples NUM_CHEMS_SES Chemicals and Side Effects and collects the CAUSES_CHcSE relationships between them.
# Returns a dataframe having columns source, target and type being the edge type
# which is in this case always CAUSES_CHcSE.
def type_nodes_from_neo4j():
    neo4j_graph = py2neo.Graph(host=host, port=port, user=user, password=password)
    nodes = neo4j_graph.run("""
    call{
    MATCH (c:Chemical) 
    with c limit """ + str(NUM_CHEMS_SES) + """
    with collect(c) as cc
    return cc
    }
    call{
    MATCH (c:SideEffect)
    with c limit """ + str(NUM_CHEMS_SES) + """
    with collect(c) as cs
    return cs
    }
    match p=(c:Chemical) -[m:CAUSES_CHcSE]- (x:SideEffect)
    WHERE c in cc and x in cs and m.avg_frequency is not null and not m.avg_frequency = 'postmarketing'
    return id(c) as source, id(x) as target, type(m) as type
    """).to_data_frame()
    return nodes


# Makes a graph object out of the given nodes.
def graph_from_nodes(nodes):
    adr_graph = StellarDiGraph(edges=nodes, edge_type_column="type")
    print(adr_graph.info())
    return adr_graph

# Turns x from a percentage string to a normal float number if it is a string or else x remains the same.
def percent(x):
    if isinstance(x, str):
        if x.rstrip('%').replace('.', '', 1).isdigit():
            return float(x.rstrip('%')) / 100.0
        else:
            return x
    else:
        return x


# If it is a string, gives x the mean value of its frequency class from means array which should have the lowest
# frequency mean of class 1 as the first entry and the rest in ascending order. For the lower frequency classes
# if there are no examples to build a mean, the arithmetic mean of the frequency class borders is used instead.
def freq_cont(x, means):
    if isinstance(x, str):
        if x == "very frequent" or x == "very common":
            return means[4]
        elif x == "frequent" or x == "common":
            return means[3]
        elif x == "infrequent" or x == "uncommon":
            if math.isnan(means[1]):
                return 0.0055
            else:
                return means[2]
        elif x == "rare":
            if math.isnan(means[1]):
                return 0.00055
            else:
                return means[1]
        elif x == "very rare":
            if math.isnan(means[0]):
                return 0.000005
            else:
                return means[0]
        else:
            return 'del'
    else:
        return x

# Puts non-string frequency values in freqs array into a list according to their respective frequency classes
# and calculates the mean for each class. Returns array of ascending frequency classes means.
def get_mean(freqs):
    classes = [[], [], [], [], []]
    for x in freqs:
        if not isinstance(x, str):
            if x >= 1 / 10:
                classes[4].append(x)
            elif x >= 1 / 100:
                classes[3].append(x)
            elif x >= 1 / 1000:
                classes[2].append(x)
            elif x >= 1 / 10000:
                classes[1].append(x)
            elif x < 1 / 100000 and x > 0:
                classes[0].append(x)
    classes_means = [np.mean(x) for x in classes]
    return classes_means

# Translates x into a frequency class type of 0 to 5. x can either be a frequency class string, a frequency float or 0.
# If x is a string that doesn't belong to any one frequency class, 'del' is given back, so that this entry is deleted
# by method delete_invalid_freqs(nodes).
def freq_class(x):
    if isinstance(x, str):
        if x == "very frequent" or x == "very common":
            return 5
        elif x == "frequent" or x == "common":
            return 4
        elif x == "infrequent" or x == "uncommon":
            return 3
        elif x == "rare":
            return 2
        elif x == "very rare":
            return 1
        else:
            return 'del'
    else:
        if x >= 1 / 10:
            return 5
        elif x >= 1 / 100:
            return 4
        elif x >= 1 / 1000:
            return 3
        elif x >= 1 / 10000:
            return 2
        elif x < 1 / 100000 and x > 0:
            return 1
        else:
            return 0

# Deletes nodes with type 'del' out of given nodes array.
def delete_invalid_freqs(nodes):
    idx = nodes.index[nodes['type'] == 'del'].tolist()
    nodes.drop(index=idx, inplace=True)


# Samples NUM_CHEMS_SES Chemicals and Side Effects and collects the CAUSES_CHcSE relationships between them.
# Either continuous frequencies (classes = False) or discrete frequency classes (classes = True) are used.
# Fills missing associations in (chemicals x side effects) matrix with zeros and returns this matrix
# along nodes array having edges and their frequencies as rows.
def pos_freq_nodes_from_neo4j(classes):
    neo4j_graph = py2neo.Graph(host=host, port=port, user=user, password=password)
    # get associations which have an average frequency values and make nodes array having source, target and type
    # column with type containing the average frequency, exclude association with average frequency 'postmarketing'
    nodes = neo4j_graph.run("""
    call{
    MATCH (c:Chemical) 
    with c limit """ + str(NUM_CHEMS_SES) + """
    with collect(c) as cc
    return cc
    }
    call{
    MATCH (c:SideEffect)
    with c limit """ + str(NUM_CHEMS_SES) + """
    with collect(c) as cs
    return cs
    }
    match p=(c:Chemical) -[m:CAUSES_CHcSE]- (x:SideEffect)
    WHERE c in cc and x in cs and m.avg_frequency is not null and not m.avg_frequency="postmarketing"
    return c.name as source, x.name as target, m.avg_frequency as type
    """).to_data_frame()

    # Turn percentages in type column into floats.
    nodes['type'] = [percent(x) for x in nodes['type']]

    # If classes boolean is True, turn frequency values in column type into frequency class values 0 to 5.
    if classes:
        nodes['type'] = [freq_class(x) for x in nodes['type']]
    else:
        # If not, translate the frequency strings like 'frequent' into the mean of the corresponding frequency values.
        freq_list = nodes['type'].values.tolist()
        means = get_mean(freq_list)
        nodes['type'] = [freq_cont(x, means) for x in nodes['type']]

    # Delete nodes that got assigned the type 'del'.
    delete_invalid_freqs(nodes)

    # Turn sampled edges into a ('source' x 'target') matrix with entries having value of 'type' column.
    matrix = pd.pivot(nodes, index='source', columns='target', values='type')
    # Fill missing values with 0s.
    matrix.fillna(0, inplace=True)

    # Bring nodes array back to its original form of 'source', 'target' and 'type' column.
    nodes = matrix.stack()
    nodes = nodes.reset_index()
    nodes.rename(columns={0: 'type'}, inplace=True)

    # nodes is returned like this to be able to perform an sklearn train_test_split() on it.
    # matrix is returned to get the column and row indices of the original ('source' x 'target') matrix
    return nodes, matrix


# Samples num_samples examples from csv file named pos and num_samples examples from csv file named neg.
# It omits the drugs Carboplatin, Oxaliplatin and Cisplatin because they cause problems in the algorithm.
# Then it shuffles the positive and negative examples and saves them in a csv file named out_name.
def shuffle(pos, neg, num_samples, out_name):
    pos = pos[pos.drug != 'Carboplatin']
    pos = pos[pos.drug != 'Oxaliplatin']
    pos = pos[pos.drug != 'Cisplatin']
    pos = pos.sample(num_samples, replace=True)

    neg = neg[neg.drug != 'Carboplatin']
    neg = neg[neg.drug != 'Oxaliplatin']
    neg = neg[neg.drug != 'Cisplatin']
    neg = neg.sample(n = num_samples, replace=True)

    pos.to_csv(out_name + '_pos.csv', index=False)
    neg.to_csv(out_name + '_neg.csv', index=False)

    out = pd.concat([pos, neg])
    out = out.sample(frac=1)
    out.to_csv(out_name + '.csv', index=False)

# Makes csv-file with num_pos_sample positive and num_pos_samples negative samples of side effect se_name being
# associated with certain chemicals and prints the results to a csv-file out_file.
def assocs_per_se(se_name, num_pos_samples, out_file):
    neo4j_graph = py2neo.Graph(host=host, port=port, user=user, password=password)
    # get chemical - side effect associations that don't contain the given side effect se_name and where
    # the smiles string of the chemical is given
    negatives = neo4j_graph.run("""
        MATCH p=(c: SideEffect)-[m]-(n:Chemical)
        where n.calculated_properties_kind_value_source is not null
        and not c.name='""" + se_name + """'
        with c.name as se, n.name as drug, m, n.calculated_properties_kind_value_source as calc_props
        call {
            with calc_props
            unwind calc_props as prop
            with split(prop, '::')[0] as prop_name, split(prop, '::')[1] as prop_val
            where prop_name = "SMILES"
            RETURN prop_val as smiles
        }
        with {drug:drug, smiles:smiles,
            freq: 0} as drugs
        return drugs.drug as drug, drugs.smiles as smiles, drugs.freq as freq
    """).to_data_frame()

    with open('data/samples' + se_name, 'w') as f:
        f.write(f'negatives: {negatives.shape[0]}\n')
    print(f'negatives: {negatives.shape[0]}')

    # get CAUSES_CHcSE relationships of given side effect to chemicals and where the smiles string
    # of the chemical is given
    positives = neo4j_graph.run("""
        match (s {name: '""" + se_name + """'})-[m:CAUSES_CHcSE]-(c:Chemical)
        where c.calculated_properties_kind_value_source is not null
        with s.name as se, c.name as drug, m, c.calculated_properties_kind_value_source as calc_props
        call {
            with calc_props
            unwind calc_props as prop
            with split(prop, '::')[0] as prop_name, split(prop, '::')[1] as prop_val
            where prop_name = "SMILES"
            RETURN prop_val as smiles
        }
        with {drug:drug, smiles:smiles,
            freq: 1} as drugs
        return drugs.drug as drug, drugs.smiles as smiles, drugs.freq as freq
    """).to_data_frame()

    with open('data/samples' + se_name, 'a') as f:
        f.write(f'positives: {positives.shape[0]}')
    print(f'positives: {positives.shape[0]}')

    shuffle(positives, negatives, num_pos_samples, out_file)



