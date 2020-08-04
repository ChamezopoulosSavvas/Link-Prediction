#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:54:38 2020

@author: Chamezopoulos Savvas
"""

import utils_n2v as utils

from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split


# filename_edges = "Datasets/Cora/cora_cites.csv"
# wt=False
filename_edges = "Datasets/soc-sign-bitcoinotc/soc-sign-bitcoinotc.csv"
wt=True

edgelist = utils.load_data(filename_edges, weighted=wt)
G = StellarGraph(edges = edgelist)
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

# get node embeddings for train graph
print("Getting node embeddings for the Train Graph...")
embedding_train = utils.node2vec_embedding(G_train, "Train Graph", weighted=wt)

binary_operators = [utils.operator_hadamard, 
                    utils.operator_l1, 
                    utils.operator_l2, 
                    utils.operator_avg]

print("Running link prediction for model selection...")
results = [utils.run_link_prediction(op, 
                                     examples_train, 
                                     labels_train, 
                                     embedding_train, 
                                     examples_model_selection, 
                                     labels_model_selection) for op in binary_operators]

best_result = max(results, key=lambda result: result["auc_score"])
print(f"Best result from '{best_result['binary_operator'].__name__}'")

# Apply best model on test data
print("\n\nApplying link prediciton with best parameters on Test graph")

# get node embeddings for train graph
print("Getting node embeddings for the Test Graph...")
embedding_test = utils.node2vec_embedding(G_test, "Test Graph", weighted=wt)

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
                                                    embedding_test,
                                                    best_result["binary_operator"])

print("Accuracy score on test set using", 
      best_result['binary_operator'].__name__,
      ":", test_acc)

