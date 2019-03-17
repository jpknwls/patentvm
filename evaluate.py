#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
eval.py: Evaluation Suite
John Knowles <jkn0wles@stanfordedu>
Sam Premutico <samprem@stanford.edu>
"""
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import numpy as np


""" -------------------------- BASELINES ------------------------  """
hidden_size = 256

def load_glove(doc_txt):
	corpus = Corpus() 
	corpus.fit(doc_txt, window=10)
	glove = Glove(no_components=hidden_size, learning_rate=0.05) 
	glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
	glove.add_dictionary(corpus.dictionary)
	glove.save('glove.model')
	glove.word_vectors[glove.dictionary['equipment']]
	return glove

def get_vectors(doc, glove):
    vecs = []
    for word in doc:
        vecs.append(glove.word_vectors[glove.dictionary[word]])
    return vecs

def avg_vecs(vecs): return np.mean(vecs, axis=0)
def max_vecs(vecs): return np.maximum.reduce(vecs)


def get_avg_glove(patent,docs, glove):
    return avg_vecs(get_vectors(docs[patent][-MAX_LENGTH:], glove))

def get_max_glove(patent,docs, glove):
    return max_vecs(get_vectors(docs[patent][-MAX_LENGTH:], glove))
          



""" ------------------------------------------------------------ """


""" -------------------- HELPER FUNCTIONS --------------------  """


def cos_sim(d1, d2):
    """


		formula:
					dot product (a, b)
				---------------------------
				  l1norm(a) *  l1norm(a) 

    """
    dp = np.dot(d1, d2)
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    return dp / (norm_d1 * norm_d2)


def inv_rank(node, labels, predictions):
    labels = set(labels)
    for index, node_id in enumerate(predictions):
        if node_id in labels:
            return float(1)/(float(index)+1)
    print("Didn't find any true neighbors in predictions for node id {}".format(node))
    return -1.0

# calculates mean reciprical rank for query set
def MRR(nodes, labels, predictions):
    Q = len(labels)
    RR = 0.0
    for i in range(Q):
        curr_rank = inv_rank(nodes[i], labels[i], predictions[i]) 
        #print(curr_rank)
        RR += curr_rank
    return (1.0/Q)*RR

def get_citations(patent, data):
    for d in data:
        if d['id'] == patent:
            return d['citations']


def get_content(patent, data):
    for d in data:
        if d['id'] == patent:
            return d['content']


def get_cluster():
    sims = [[cos_sim(f, algo(p, docs)), p] for p in all_patents if p != patent]
    


""" ------------------------------------------------------------ """



def evaluate(data, patents, all_patents, docs, doc_txt, k):

    
    glove = load_glove(doc_txt)

    baselines = [get_avg_glove, get_max_glove]


    MRRs = []

    #BASELINES

    for baseline in baselines:
        nodes = []
        labels = []
        predictions = []
        for patent in patents:
            citations = get_citations(patent, data)
            f = baseline(patent, docs, glove)  
            sims = [[cos_sim(f, algo(p, docs)), p] for p in all_patents if p != patent]
            top_k = np.argsort([s[0] for s in sims])[-k:]
            top_patents = [sims[i][1] for i in top_k]     
            nodes.append(patent)
            labels.append(citations)
            predictions.append(top_patents)
        MRRs.append(MRR(nodes, labels, predictions))
    


    nodes = []
    labels = []
    predictions = []

    for patent in patents:

        citations = get_citations(patent, data)
        p_content = get_content(patent, data)

        sims = 

        top_k = np.argsort(predicter())
        top_patents = [sims[i][1] for i in top_k]     
        nodes.append(patent)
        labels.append(citations)
        predictions.append(top_patents)



    return MRRs
    

# toy example from https://en.wikipedia.org/wiki/Mean_reciprocal_rank
def test_toy():
	CATS = "cats"
	TORI = "tori"
	VIRUSES = "viruses"

	q1 = ["catten", "cati",CATS]
	q2 = ["torii", TORI, "toruses"]
	q3 = [VIRUSES, "virii", "viri"]
	predictions = [q1,q2,q3]
	labels = [[CATS],[TORI],[VIRUSES]]
	nodes = [CATS, TORI, VIRUSES]
	mrr = MRR(nodes, labels, predictions)
	print("MRR: {}".format(mrr))


def test_full(): return

	# iterate over all documents
	# build a list of the top k docs in 'cluster'


