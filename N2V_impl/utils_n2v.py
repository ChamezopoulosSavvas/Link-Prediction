#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:23:32 2020

@author: Chamezopoulos Savvas
"""

import networkx as nx
import numpy as np
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import multiprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def load_data(filename, weighted=False):
    
    if weighted == False:
        #validate filename and read graph
        try:
            G = nx.read_edgelist(filename, delimiter=',')
            # G = nx.fast_gnp_random_graph(100, 
            #                              0.3, 
            #                              seed=None, 
            #                              directed=False)
            # print(nx.info(G))
        except:
            print("Please enter a valid filename")
            return 0
        
        edgelist_df = nx.to_pandas_edgelist(G)
        edgelist_df.columns = ['source', 'target']
    
    else:
        #validate filename and read graph
        try:
            G = nx.read_weighted_edgelist(filename, delimiter=',')
            # G = nx.fast_gnp_random_graph(100, 
            #                              0.3, 
            #                              seed=None, 
            #                              directed=False)
            # print(nx.info(G))
        except:
            print("Please enter a valid filename")
            return 0
        
        edgelist_df = nx.to_pandas_edgelist(G)
        edgelist_df.columns = ['source', 'target','weight']
        
    #return G
    return edgelist_df

# Produces the node embeddings to be used by the ml
def node2vec_embedding(graph, name, weighted=False):
    
    p = 1.0
    q = 1.0
    dimensions = 128
    num_walks = 10
    walk_length = 80
    window_size = 10
    num_iter = 1
    workers = multiprocessing.cpu_count()
    
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), 
                   n=num_walks, 
                   length=walk_length, 
                   p=p, 
                   q=q, 
                   weighted=weighted)
    
    print(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0,
                     sg=1, workers=workers, iter=num_iter) 
    
    def get_embedding(u):
        return model.wv[u]

    return get_embedding


# link embeddings
# Returns the edge embedding from the nodes embedding as produced
# by the binary operator.
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [binary_operator(transform_node(src), transform_node(dst))
            for src, dst in link_examples]

# training classifier
# returns the classifier fitted on the edges provided
def train_link_prediction_model(link_examples, 
                                link_labels, 
                                get_embedding, 
                                binary_operator):
    
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(link_examples, 
                                              get_embedding, 
                                              binary_operator)
    
    clf.fit(link_features, link_labels)
    return clf


# returns the classifier preprocessed so the data have 0 mean and unti variance
# to do so, sklearn classes Pipeline and StandardScaler are used
def link_prediction_classifier(max_iter=2000):
    
    lr_clf = LogisticRegressionCV(Cs=10, 
                                  cv=10, 
                                  scoring="roc_auc", 
                                  max_iter=max_iter)
    
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


# evaluate classifier
# returns the auc score for the test data
def evaluate_link_prediction_model(clf, 
                                   link_examples_test, 
                                   link_labels_test, 
                                   get_embedding, 
                                   binary_operator):
    
    link_features_test = link_examples_to_features(link_examples_test, 
                                                   get_embedding, 
                                                   binary_operator)
    
    score_roc, score_acc, score_f1, score_rec, score_pres = evaluate_model(clf, 
                                                    link_features_test, 
                                                    link_labels_test)
    return score_roc, score_acc, score_f1, score_rec, score_pres


# applies the classifier on the test data and returns the auc score
def evaluate_model(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    auc = roc_auc_score(link_labels, predicted[:, positive_column])
    
    predicted = clf.predict(link_features)
    
    fpr, tpr, _ = roc_curve(link_labels, predicted)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    acc = accuracy_score(link_labels, predicted)
    f1 = f1_score(link_labels, predicted)
    rec = recall_score(link_labels, predicted)
    pres = precision_score(link_labels, predicted)
    
    return auc, acc, f1, rec, pres

# this function applies the link prediction steps using the apove functions
# firstly a classifier is produced with training data and operator specified from the
# arguments, and then applies it on test data and returns an auc score
def run_link_prediction(binary_operator, 
                        examples_train, 
                        labels_train, 
                        embedding_train, 
                        examples_model_selection, 
                        labels_model_selection):
    
    clf = train_link_prediction_model(examples_train, 
                                      labels_train, 
                                      embedding_train, 
                                      binary_operator)
    
    score_auc, score_acc, score_f1, score_rec, score_pres = evaluate_link_prediction_model(clf,
                                           examples_model_selection,
                                           labels_model_selection,
                                           embedding_train,
                                           binary_operator)

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "auc_score": score_auc,
        "acc_score": score_acc,
        "f1_score": score_f1,
        "precision_score": score_pres,
        "recall_score": score_rec
    }

# below follow 4 support functions that implement 4 possible operators to
# produce an edge embedding from 2 node embeddings
def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0


