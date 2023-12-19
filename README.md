# How to run ADR-prediction algorithms


**Prerequisits**

* Neo4j Version 4.4.8

To run ComplEx algorithm or matrix decomposition first start Neo4j. For deep learning when creating data sets to be used as input to the algorithm with method `make_data_sets()` you also need to first start Neo4j.


**General**

To adjust the amount of input data change the variables `NUM_CHEMS_SES` or `LIMIT_CLASS_0` in file neograph.py.
 - `NUM_CHEMS_SES`: number of chemicals and side-effects to sample
  - `LIMIT_CLASS_0`: number of unassociated chemicals and side-effects
   pairs to sample (only needed for categorical)

Result files can be found in the data folder of the individual algorithm.


**Run ComplEx**

Run either binary.py or categorical.py if you want to predict the existance of certain drug-side effect relations or their frequency classes respectively.

The fractions of the data used for training, testing and validation as well as the number of cross-validation splits and iterations for hyperparameter optimization can be changed through the global variables of binary.py or categorical.py respectively.


**Run deep learning**

Enter the name of the data set you want to use in the `name` variable in dictionary `params`.
You can create additional data sets by defining a new dictionary within `dataset_info` dictionary.

For example:

    derm = dict(  
            nll_func = binary_classification_nll,  
            target_name = 'freq', 
            data_file ='dermatitis_perioral')

 - `nll_func`: negative log likelyhood  function to use 
 - `target_name`: name of the target column in the given CSV file 
 - `data_file`: CSV file with a column with the smiles strings called 'smiles' and a target
   column holding 1's and 0's, where 1 stands for a detection of the
   target property (here the association of this drug to the given side
   effect) and 0 stands for no detection of the target

To make input files you can also use the method `neo_graph.assocs_per_se` and give the name of a side effect in PharMeBINET, the number of positive and negative examples to sample and a file name for the CSV file to be created.

    neo_graph.assocs_per_se([side effect], [number of positive / negative samples], [filename])

When you run this method, Neo4j must be running at the same time.

To run the deep learning algorithm, run main.py in the `deep_learning` folder.


**Run Matrix Decompostition**

To perform matrix decomposition run main.py in `matrix_decomp` folder.
There are different options for evaluation. Regression metrics are calculated by method `final_eval_regression`, categorical metrics by `final_eval_categorization` and binary ones by `final_eval_binary`.
Continuous (in `train_cont` and `train_cont`) or categorical frequency values (in `train_classes` and `test_classes`) can be used. The resulting evaluation files in the data folder have a specified extension, for example  '_classes' or '_cont'. `final_eval_binary` needs to be given binary thresholds that can either be optimized for the categorical or the continous data. Optimal binary thresholds can be produced by the method `thresholds.binary_thresh([training set], zeros, k, alpha, [number of cross-validation splits])`.  

Run `randSearchExp([number of cross-validation splits], [number of iterations], zeros, [training set])` for hyperparameter optimization. It will print out a line of the hyperparameters producing the pareto optimal pairs of RMSE and PR AUC. A line with PR AUC values and a line with RMSE values. One can then choose the desired hyperparameter combination and set them as global variables. Then run the final evaluation.
