"""
TAL² 
CPES SDAC
2025-2026
"""
from random import randint, seed
from os import scandir
import torch
from torch import Tensor, nn, cat, zeros
from tqdm import tqdm, trange
import numpy as np

"""
Dans le TP de cette semaine on va faire de la prédiction de langue à partir de texte en utilisant des réseaux de neuronnes : MLP, CNN et RNN.

Comme d'habitude, les premières lignes de code servent à charger les données et à initialiser des variables.
"""


# la taille des jeux de données d'entrainement et de test par langue
K = 500
LEN = 20 # la longeur des chaine de caractères que l'on va essayer de classifier

# on initialise le generateur de nombres aléatoires
seed(0)


# reading the data
path = 'data/'
data = {}
for fl in scandir(path):
    if fl.is_file() and   '.' not in fl.name and '~' not in fl.name:
        with open(fl) as f:
            text = ''
            for l in f:
                l = l.strip()
                if l == '':
                    continue
                text += l

            data[fl.name] = text

if len(data) == 0:
    print('You need to edit the path variable to match the place where the data are stored.')
    exit()

"""
data est un dictionnaire avec en clé des iso de langues et en valeur des textes dans ces langues.
"""

# get the characters, and the languages

# Truly, there are libraries to do all of this, but learning to do them by hand is good, you understand what you do
chars = set()
for lng, text in data.items():
    #print(x, len(y))
    chars.update(text)

chars = sorted(chars)
ioc = {c:i for i,c in enumerate(chars)} # that's a home made character based tokenizer !
nchar = len(chars)

#lngs = sorted(data)
lngs = ['als', 'de', 'en', 'fy', 'nl',
        'br', 'cy', 'ga', 'gd', 'gv',	
        'ca', 'es', 'fur', 'it', 'pt',
        'mg', 'sw', 'xh', 'yo', 'zu']
iol = {l:i for i,l in enumerate(lngs)} # turns a language iso into an int
nclass = len(iol)


# train, test, split
train = {x:y[:-500] for x,y in data.items()} # on va prendre les premiers 500 caractères de chaque langue comme jeu d'entrainement
test = {x:y[-500:] for x,y in data.items()} # le reste comme test




# let's prepare a standardize train test so that we can compare the results across models
XY_train = []
for _ in trange(K):
    for lng, text in train.items():
        # pick a random position is the text
        r = randint(0, len(text) - LEN)

        XY_train.append((text[r:r+LEN], lng))


XY_test = []
for _ in trange(K):
    for lng, text in test.items():
        # pick a random position is the text
        r = randint(0, len(text) - LEN)

        XY_test.append((text[r:r+LEN], lng))




# things we can do with raw text data:
# predict the language from a sequence of characters



# to train a simple neural network
# you need :

# a model :
def perceptron(n_char, n_class):
    return nn.Sequential(nn.Linear(n_char, n_class)) # une seule couche linéaire, pas de non linéarité


model = perceptron(nchar, nclass)      # a model
floss = nn.CrossEntropyLoss()   # a loss function, la crossEntropy est standard pour les tâches de classif
trainer = torch.optim.Adam(model.parameters())    # an optimizer, c'est ça qui fait le pas de gradient, en fait il y a mieux que la descende de gradient brut, ici on prend adam, mais vous pouvez jouer avec d'autres, check torch.optim

# some data, that we have already read

# and a training loop
for txt, lng in tqdm(XY_train):
    # prepare the data
    # in practice you'd prepare the data much before you train in a separate loop
    y = iol[lng] # the target
    x = [0] * nchar # the input, a bag of characters
    for c in txt:
        x[ioc[c]] += 1

    #print(x, y)
    
    # empty the gradient
    trainer.zero_grad()
    
    #compute the score for each langage
    scores = model(Tensor([x]))
    #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')

    # compute the loss according the true language y
    loss = floss(scores, Tensor([y]).long())
    
    # compute the gradient by back propagation
    loss.backward()
    
    # make a step in the direction of the gradient
    trainer.step()

    # repeat

# once we're done : we test

# test
trainer.zero_grad()
confusion = np.zeros((nclass, nclass))

"""
YOUR TURN

write the test loop and fill a confusion matrix
"""

print('Perc')

print('..', '\t'.join(lngs), sep='\t')
for i,lng in enumerate(lngs):
    print(lng, *confusion[i], sum(confusion[i]), sep='\t')

print('..', *[sum(confusion[:,i]) for i in range(len(lngs))], sep='\t')
print('Acc: ', np.trace(confusion) / confusion.sum())

print()

# Analyse the above code and try to understand what is happening


"""
YOUR TURN

take inspiration from the simple perceptron and create a multi layer perceptron.
Train it, test it, on the same data and compare them.
"""






"""
THEN :

modify your mlp, change the depth/width of the MLP layers, change the non linearities, train for longer, add dropout ...
"""





# then look at the CNN code below, try to understand it and see if it works better than an MLP


transp = nn.Module()
transp.forward = lambda x: x.transpose(1,2) # simply overriding the forward method does the job, sweet!!!

flat = nn.Module()
flat.forward = lambda x: x[0] # RNN layers also return funky things, here we just remove the outer layer of a [[list]]


def cnn(nchar, char_dim, n_class):
    return nn.Sequential(nn.Embedding(nchar, char_dim),
                         transp,
                         nn.Conv1d(char_dim, 50, 3),
                         nn.ReLU(),
                         nn.Conv1d(50, 50, 3),
                         nn.ReLU(),
                         nn.AdaptiveMaxPool1d(1),
                         transp,
                         nn.Linear(50, n_class),
                         flat)


# train a CNN, note that now the input is not a bag of characters but a list of indices that we'll be turned into embeddings

for txt, lng in tqdm(XY_train):
    # prepare the data
    y = iol[lng]
    x = [ioc[c] for c in txt] # this is not a vector of count, but a sequence of indices

    """
    YOUR TURN
    
    train a model and test it, don't forget the confusion matrix

    """

print()




"""
YOUR TURN

Take inspiration from the CNN to try to implement an RNN
"""

# the following function can be useful
caps = nn.Module()
caps.forward = lambda x: x[0][:,-1]
