#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:28:15 2020

@author: chamezos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:54:38 2020

@author: Chamezopoulos Savvas
"""

import utils_ctdne as utils

from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split

# import data and create initial graph G
# filename_edges = "Datasets/Cora/cora_cites.csv"
# wt = False
filename_edges = "Datasets/soc-sign-bitcoinotc/soc-sign-bitcoinotc-temporal.csv"
wt=True

edgelist_df = utils.load_data(filename_edges, weighted = wt)

G = StellarGraph(edges = edgelist_df)
print("\n", G.info())
print("Created master graph from data")

# Define an edge splitter on the original graph:
edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, 
# and same number of negative links, from graph, and obtain the
# reduced graph graph_test with the sampled links removed:
(
 G_test,         # To compute node embeddings with mode edges than G_train
 examples_test,  
 labels_test
 ) = edge_splitter_test.train_test_split(p=0.1, 
                                         method="global")

#print(G_test.info())
print("Created test Graph from master graph")

# Do the same process to compute a training subset from within the test graph
edge_splitter_train = EdgeSplitter(G_test, G)
(
 G_train,      # To compute node embeddings
 examples, 
 labels 
 ) = edge_splitter_train.train_test_split(p=0.1, 
                                          method="global")
print("Created train Graph from test graph")
                                          
                                          
(
 examples_train,            # For training classifiers. They dont exist in G_train
 examples_model_selection,  # For choosing the best classifier
 labels_train,              # For training classifiers. They dont exist in G_train
 labels_model_selection,    # For choosing the best classifier
) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

#print(G_train.info())
print("Extraced train and test data from train graph for model selection")

# get temporal node embeddings for train graph
print("Getting temporal node embeddings for the Train Graph...")
temporal_embedding_train = utils.ctdne_embedding(G_train, "Train Graph")

binary_operators = [utils.operator_hadamard, 
                    utils.operator_l1, 
                    utils.operator_l2, 
                    utils.operator_avg]

print("Running link prediction for model selection...")
results = [utils.run_link_prediction(op, 
                                     examples_train, 
                                     labels_train, 
                                     temporal_embedding_train, 
                                     examples_model_selection, 
                                     labels_model_selection) for op in binary_operators]

best_result = max(results, key=lambda result: result["auc_score"])
print(f"Best result from '{best_result['binary_operator'].__name__}'")

# Apply best model on test data
print("\n\nApplying link prediciton with best parameters on Test graph")

# get node embeddings for train graph
print("Getting temporal node embeddings for the Test Graph...")
temporal_embedding_test = utils.ctdne_embedding(G_test, "Test Graph")

print("Running link prediciton on test data...")
(
 test_auc, 
 test_acc, 
 test_f1,
 test_rec,
 test_pres
) = utils.evaluate_link_prediction_model(best_result["classifier"],
                                                    examples_test,
                                                    labels_test,
                                                    temporal_embedding_test,
                                                    best_result["binary_operator"])

print("ROC AUC score on test set using", 
      best_result['binary_operator'].__name__,
      ":", test_auc)


print("\n\nApplying STATIC node embedding on same test data for comparison...")

# recreate graph w/o timestamps
if wt == True:
    edgelist_df = edgelist_df[['source','target']]
    G = StellarGraph(edges = edgelist_df)
    print("\n", G.info())
    print("Created Static graph with no edge weights")
    edge_splitter_test = EdgeSplitter(G)
    (
     G_test,         
     examples_test,  
     labels_test
     ) = edge_splitter_test.train_test_split(p=0.1, 
                                             method="global")


# get node embeddings for test graph
print("Getting static node embeddings for the Test Graph...")
static_embedding_test = utils.node2vec_embedding(G_test, "Test Graph")

print("Running link prediciton on test data...")
(
 test_auc_static, 
 test_acc_static, 
 test_f1_static,
 test_rec_static,
 test_pres_static
) = utils.evaluate_link_prediction_model(best_result["classifier"],
                                                    examples_test,
                                                    labels_test,
                                                    static_embedding_test,
                                                    best_result["binary_operator"])

print("ROC AUC score on test set using", 
      best_result['binary_operator'].__name__,
      ":", test_auc_static)




