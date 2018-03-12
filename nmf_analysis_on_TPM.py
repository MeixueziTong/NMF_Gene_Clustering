#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:41:59 2018

@author: meixuezi
"""
# Apply None-negative Matrix Factorization (NMF) to RNA-seq data (e.g. a TPM gene expression table of n genes * m samples )
# to find genes with similar expression pattern with the gene of interest

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from time import time

print('read data...')

TPM = pd.read_table('TPM.txt', sep = '\t') # import TPM table
npArrayTPM = np.array(TPM)
gene_names = npArrayTPM[:,0] # e.g. AT1G00010
elf3 = pd.read_table('RNA-seq_elf3-1_timecourse_combined_20140901_TPM.txt', sep = '\t') # obtain gene short name from another file
gene_short_names = np.array(elf3)[:,1] # e.g. HSP70


scaler = MaxAbsScaler()
normalizer = Normalizer() # for computing cosine similarity
n_components = 40 # specify the number of components
nmf = NMF(n_components) 

pipeline = make_pipeline(scaler, nmf, normalizer)

print('compute data...')
tick = time()

norm_features = pipeline.fit_transform(npArrayTPM[:,1:]) # take the index column off

df = pd.DataFrame(norm_features, index = gene_short_names) # get a table of feature numbers for each gene shape = [33557, 40]


gene_list = ['LFY','SEP3','SRL2','ELF4'] # gene of interest

similar_genes = {}
columns = []      
for gene in gene_list:

    gene_index = df.loc[gene] #  get feature values for 'gene', gene_index.shape = (40,) 
    similarities = df.dot(gene_index) # [33557, 40]*[40, 1] = [33557,1] cosine similarity: normalized dot product (inner product)
    #print('Top 50 similar genes for %s' % gene)
    top50 = similarities.nlargest(50)
    #print(similarities.nlargest(50))
    similar_genes[str(gene)] = top50.index
    columns.append(str(gene))
    similar_genes['similarity score to '+ str(gene)] = top50.values
    columns.append('similarity score to '+ str(gene))

similar_genes = pd.DataFrame(similar_genes, columns = columns)
tock = time()
print('computing done in %0.2fs!!'%(tock - tick))
print('write data into file...')
similar_genes.to_csv('similar_gene_list.csv', index = False)

print('job done!!')

    