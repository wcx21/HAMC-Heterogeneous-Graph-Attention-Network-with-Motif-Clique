from __future__ import print_function
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import json
import random
import subprocess
from sklearn.preprocessing import MultiLabelBinarizer
from metrics import *


def preprocess_adj_zero_one(adj):
    '''Preprocess adjacency matrix for Motif motif-cnn model and convert to tuple representation.
       Return - A list of normalized motif adjacency matrices in tuple format.
                Shape: (n_motifs, (coords, values, mat_shape)), mat_shape should be of (n_positions, N, N)'''
    normalized_adjs = []
    for m in range(0, len(adj)):
        normalized_adj_m = []
        normalized_adj_m.append(sp.eye(adj[m][0].shape[0]))
        # normalized_adj_m.append(sp.eye(adj[m][0].shape[0]))
        for k in range(0, len(adj[m])):
            # adj_normalized = normalize_adj(adj[m][k])
            adj_normalized = adj[m][k]
            # print(adj_normalized)
            adj_normalized.data = np.array([1.] * len(adj_normalized.data))
            normalized_adj_m.append(adj_normalized)
            print(adj_normalized.data.shape)
        normalized_adjs.append(normalized_adj_m)

    normalized_adjs = sparse_to_tuple(normalized_adjs)
    return normalized_adjs


def discover_motif_clique(adj, num=1):
    '''
    Discovering target-target link which is equivalent to m-clique
    Return - A new adj list.
    '''
    m_clique_adj = adj
    n_nodes = adj[0][0].shape[0]
    # for m in range(0, len(adj)):
    for m in range(0, num):
        temp = adj[m][0].tocoo()
        col = temp.col
        row = temp.row
        a_list = dict()
        for i in range(len(col)):
            c = col[i]
            if c not in a_list.keys():
                a_list.update({c: [row[i]]})
            else:
                a_list[c].append(row[i])
        new_index = [[], []]
        for v in a_list.values():
            for i in range(len(v)):
                for j in range(len(v)):
                    # if i == j:
                    #     continue
                    new_index[0].append(v[i])
                    new_index[1].append(v[j])
        # print(len(new_index[0]))
        tar2tar_mat = sp.csr_matrix((np.ones(len(new_index[0]), dtype=np.float32), (new_index[0], new_index[1])), shape=(n_nodes, n_nodes))
        m_clique_adj[m] = [tar2tar_mat] + adj[m]
    return m_clique_adj


def no_features(n_nodes):
    return sp.eye(n_nodes).tocsr()
