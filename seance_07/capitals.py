from os import scandir
from tqdm import tqdm
from numpy import zeros
from numpy.linalg import norm
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from math import log
from collections import defaultdict


# LOAD THE DATA
data = {}
fls = [x.name for x in scandir('pages') if '~' not in x.name] # the list of pages
fls.sort()

for fname in fls: # for each page
    data[fname] = []
    fl = open('pages/'+fname) # open it # in windows/mac you may need to set encoding='utf-8'

    for l in fl: # read line by line
        l = l.strip()
        if l == '':
            continue

        data[fname].append(l.split())
    #print(fname, len(data[fname]))


capitals = ['amsterdam', 'athenes', 'berlin', 'bruxelles', 'copenhague', 'helsinki',
            'lisbone', 'ljubljana', 'madrid', 'paris', 'prague', 'riga',
            'rome', 'stockholm', 'tallinn', 'varsovie', 'beijing', 'seoul',
            'budapest']


# extract the vocabulary
voc = defaultdict(int)
for fname, content in data.items():
    for line in content:
        for w in line:
            voc[w] += 1


print('Size voc before threshold :', len(voc))
voc = sorted([k for k, n in voc.items() if n > 25]) # YOU CAN VARY THIS
voc = sorted(voc)
print('After :', len(voc))


# create a pair of dictionaries : int of file and int of word, that are used as reference indices
iof = {fname:i for i, fname in enumerate(sorted(fls))}
iow = {w:i for i,w in enumerate(voc)}

colors = ['b' if f in capitals else 'g' for f in iof]

# now the matrix
vecs = zeros((len(data), len(voc)))

for fname, content in data.items():
    # I
    # YOUR CODE HERE
    
# now reduce dimension II
# YOUR CODE HERE


# save a figure from the reduced dimensions with pyplot
def save_redux(X, fname):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=colors) # color the points according to whether documents speak of a capital or not

    for i, fn in enumerate(fls):
        ax.annotate(fn, X[i])

    fig.set_size_inches(10, 10)
    plt.savefig(fname)

save_redux(X, 'PCA_raw')


# tsne III
# YOUR CODE HERE

save_redux(X, 'tsne-cosine_raw')



# now normalise
# TF IDF

for i, v in enumerate(vecs):
    t = sum(v)
    for j, k in enumerate(v):
        if k == 0:
            continue
        vecs[i,j] = k / t * log(len(data) / len([x for x in vecs[:,j] if x != 0]))




# PCA again V
# YOUR CODE HERE

save_redux(X, 'PCA_tfidf')


# tsne again
# YOUR CODE HERE

save_redux(X, 'tsne-cosine_tfidf')
