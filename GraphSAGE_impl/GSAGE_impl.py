#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:23:19 2020

@author: Chamezopoulos Savvas
"""

import utils_sage as utils
import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification

import tensorflow.keras.metrics as tfm
from tensorflow import keras

# import data and create initial graph G
# filename_edges = "Datasets/Cora/cora_cites.csv"
# filename_features = "Datasets/Cora/cora_content.csv"
# ft = True
filename_edges = "Datasets/soc-sign-bitcoinotc/soc-sign-bitcoinotc.csv"
filename_features = ""
ft = False

edgelist_df, node_features_df = utils.load_data(filename_edges, 
                                                filename_features,
                                                features=ft)

G = StellarGraph(node_features_df, edgelist_df)
print("Created master graph from data")
print("\n", G.info())

# # Reduce initial Graph
# edge_splitter_test = EdgeSplitter(G)

# G,_,_ = edge_splitter_test.train_test_split(p=0.8, 
#                                             method="global", 
#                                             keep_connected=True)                                     
# print("\n", G.info())

# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
(
 G_test, 
 edge_ids_test, 
 edge_labels_test
) = edge_splitter_test.train_test_split(p=0.1, 
                                        method="global", 
                                        keep_connected=True)


                                        # Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
(
 G_train, 
 edge_ids_train, 
 edge_labels_train
) = edge_splitter_train.train_test_split(p=0.1, 
                                         method="global", 
                                         keep_connected=True)
                                         
print(G_train.info())
print(G_test.info())

batch_size = 20

num_samples = [20, 10]

train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

layer_sizes = [20, 20]
graphsage = GraphSAGE(layer_sizes=layer_sizes, 
                      generator=train_gen, 
                      bias=True, 
                      dropout=0.3)

# Build the model and expose input and output sockets of graphsage model
# for link prediction
x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(output_dim=1, 
                                 output_act="relu", 
                                 edge_embedding_method="ip")(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
              loss=keras.losses.binary_crossentropy,
              metrics=[tfm.AUC(),
                       "acc",
                       utils.get_f1,
                       utils.get_pres,
                       utils.get_rec])

init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
    
epochs = 20

history = model.fit(train_flow, 
                    epochs=epochs, 
                    validation_data=test_flow, 
                    verbose=2) 

sg.utils.plot_history(history)



train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
