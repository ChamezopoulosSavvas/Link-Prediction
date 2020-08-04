#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:23:32 2020

@author: Chamezopoulos Savvas
"""

import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import multiprocessing
import tensorflow.keras.backend as K

def load_data(filename_edges, filename_features, features=False):

    #validate filename and read graph
    try:
        # G = nx.read_edgelist(filename_edges, delimiter=',')
        G = nx.read_weighted_edgelist(filename_edges, delimiter=',')
        #G = nx.fast_gnp_random_graph(100, 0.3, seed=None, directed=False)
        #print(nx.info(G))
    except:
        print("Please enter a valid filename")
        return 0
    
    edgelist_df = nx.to_pandas_edgelist(G)
    # edgelist_df.columns = ['source', 'target']
    edgelist_df.columns = ['source', 'target', 'weight']
    
    if features == False:
        
        print("No features provided, constructing features using node2vec...")
        node_features_df = node2vec_embedding(G)
       
    else:
        
        print("Loading features from file:", filename_features)
        try:
            
            node_features_df = pd.read_csv(filename_features, 
                                           delimiter = '\t',
                                           header = None)
            
            node_features_df = node_features_df.fillna(0)
            node_features_df = node_features_df.iloc[:, :-1]
            node_features_df.set_index(0, inplace = True)
            node_features_df.index = node_features_df.index.map(str)
            
        except:
            print("Please enter a valid filename for the features file")
            
            node_features_df = pd.DataFrame(np.ones((len(list(G.nodes())),5)), 
                                        index = list(G.nodes))
            
    #return G
    return edgelist_df, node_features_df


# Produces the node embeddings to be used as features
def node2vec_embedding(graph):
    
    p = 1.0
    q = 1.0
    dimensions = 128
    num_walks = 10
    walk_length = 80
    window_size = 10
    num_iter = 1
    workers = multiprocessing.cpu_count()
    
    graph = StellarGraph(graph)
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
    
    print(f"Number of random walks: {len(walks)}")

    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0,
                     sg=1, workers=workers, iter=num_iter) 
    
    features = pd.DataFrame(data = model.wv.vectors, index = list(graph.nodes()))
    features.index = features.index.map(str)
    
    return features   


def get_f1(y_true, y_pred): #taken from old keras source code
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    
    return f1_val

def get_pres(y_true, y_pred): #taken from old keras source code
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    
    return precision

def get_rec(y_true, y_pred): #taken from old keras source code
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    
    return recall