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

Correction
"""

K = 500
LEN = 20


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
train = {x:y[:-500] for x,y in data.items()}
test = {x:y[-500:] for x,y in data.items()}






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


# test
trainer.zero_grad()
confusion = np.zeros((nclass, nclass))

for text, lng in tqdm(XY_test):
    # prepare the data
    y = iol[lng]
    x = [0] * nchar
    for c in text:
        x[ioc[c]] += 1


    #print(x, y)
    # get the scores
    scores = model(Tensor([x]))
    # take the argmax
    yhat = scores.argmax().item()
    #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
    # do they match or not ?
    confusion[y, yhat] += 1


def print_confusion(confusion, lngs):
    """
    make it a function to save code later
    """
    
    print('..', '\t'.join(lngs), sep='\t')
    for i,lng in enumerate(lngs):
        print(lng, *confusion[i], sum(confusion[i]), sep='\t')

    print('..', *[sum(confusion[:,i]) for i in range(len(lngs))], sep='\t')
    print('Acc: ', np.trace(confusion) / confusion.sum())


print('Perceptron')
print_confusion(confusion, lngs)
    
    


# train an mlp, it's like a perceptron but with more layers
# you need :

def mlp(nchar, n_class): # change the hidden dimensions and the non linearities : https://docs.pytorch.org/docs/2.9/nn.html
    return nn.Sequential(nn.Linear(nchar, 50),
                         nn.ReLU(),
                         nn.Linear(50, 30),
                         nn.ReLU(),
                         nn.Linear(30, n_class))


model = mlp(nchar, nclass)      # a model
floss = nn.CrossEntropyLoss()   # a loss function 
trainer = torch.optim.Adam(model.parameters())    # an optimizer

for txt, lng in tqdm(XY_train):
    # prepare the data
    # in practice you'd prepare the data much before you train in a separate loop
    y = iol[lng]
    x = [0] * nchar
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
    # compute the gradient
    loss.backward()
    # make a step
    trainer.step()


# test
trainer.zero_grad()
confusion = np.zeros((nclass, nclass))

for text, lng in tqdm(XY_test):
    # prepare the data
    y = iol[lng]
    x = [0] * nchar
    for c in text:
        x[ioc[c]] += 1


    #print(x, y)
    # get the scores
    scores = model(Tensor([x]))
    # take the argmax
    yhat = scores.argmax().item()
    #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
    # do they match or not ?
    confusion[y, yhat] += 1

print('MLP')
print_confusion(confusion, lngs)

print()

# Analyse the above code and try to understand what is happening
# modify it, increase the size of the windows we want to classify, change the depth/width of the MLP, change the non linearities, train for longer...
# then look at the CNN try to understand it and see if it works better
# eventually, try to implement an RNN∕LSTM∕GRU in the same fashion







# train a CNN

# for memory and computation reasons, convolution layers prefer to receive their input in a different order
# thus i created a simple nn layer that transposes the last too dimensions of a tensor
# to understand convolution : https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md


transp = nn.Module()
transp.forward = lambda x: x.transpose(1,2) # simply overriding the forward method does the job, sweet!!!

flat = nn.Module()
flat.forward = lambda x: x[0] # RNN layers also return funky things, here we just remove the outer layer of a [[list]]


def cnn(nchar, char_dim, n_class):
    return nn.Sequential(nn.Embedding(nchar, char_dim),   # embed the imput : index to vectors
                         transp,                          # transpose
                         nn.Conv1d(char_dim, 50, 3),      # first convolution layer, reads 3 vectors of char_dim dimensions and output a single vector of 50 dimensions
                         nn.ReLU(),                       # non linearity
                         nn.Conv1d(50, 50, 3),            # one more convolution
                         nn.ReLU(),                       # non linearity
                         nn.AdaptiveMaxPool1d(1),         # max pooling to one value per dimensions
                         transp,                          # transpose again
                         nn.Linear(50, n_class),          # final linear layer to produce the scores
                         flat)                            # since we work on the single item, remove the unnecessary batch dimension



model = cnn(nchar, 50, nclass)      # a model
floss = nn.CrossEntropyLoss()   # a loss function 
trainer = torch.optim.Adam(model.parameters())    # an optimizer

for txt, lng in tqdm(XY_train):
    # prepare the data
    y = iol[lng]
    x = [ioc[c] for c in txt] # this is not a vector of count, but a sequence of indices


    #print(x, y)
    # empty the gradient
    trainer.zero_grad()
    #compute the score for each langage
    scores = model(Tensor([x]).long())
    #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
    # compute the loss according the true language y
    #print(scores)
    loss = floss(scores, Tensor([y]).long())
    # compute the gradient
    loss.backward()
    # make a step
    trainer.step()


# test
trainer.zero_grad()
confusion = np.zeros((nclass, nclass))

for text, lng in tqdm(XY_test):
    # prepare the data
    y = iol[lng]
    x = [ioc[c] for c in text] # this is not a vector of count, but a sequence of indices

    #print(x, y)
    # get the scores
    scores = model(Tensor([x]).long())
    # take the argmax
    yhat = scores.argmax().item()
    #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
    # do they match or not ?
    confusion[y, yhat] += 1

print('CNN')
print_confusion(confusion, lngs)
print()







# train a RNN
# you need :

caps = nn.Module()
caps.forward = lambda x: x[0][:,-1] # RNN layers also return funky things

def rnn(nchar, char_dim, n_class):
    return nn.Sequential(nn.Embedding(nchar, char_dim),             # embedding 
                         nn.GRU(char_dim, 50, 1, batch_first=True), # straight into a recurrent layer
                         caps,                                      # recurrent layers return many things, we only need the final representation
                         nn.Linear(50, n_class))                    # final linear layer to compute scores

model = rnn(nchar, 50, nclass)      # a model
floss = nn.CrossEntropyLoss()   # a loss function 
trainer = torch.optim.Adam(model.parameters())    # an optimizer

# train the RNN for 5 epochs check what happens
# contrary to what we have done so far, you hardly ever trai for a single loop, you train for several, we call a loop an epoch

for epoch in range(5): 
    for txt, lng in tqdm(XY_train):
        # prepare the data
        y = iol[lng]
        x = [ioc[c] for c in txt] # this is not a vector of count, but a sequence of indices, same as for the CNN
        

        #print(x, y)
        # empty the gradient
        trainer.zero_grad()
        #compute the score for each langage
        scores = model(Tensor([x]).long())
        #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
        # compute the loss according the true language y
        #print(scores)
        loss = floss(scores, Tensor([y]).long())
        # compute the gradient
        loss.backward()
        # make a step
        trainer.step()
        
        
    # test
    trainer.zero_grad()
    confusion = np.zeros((nclass, nclass))
    
    for text, lng in tqdm(XY_test):
        trainer.zero_grad()
        # prepare the data
        y = iol[lng]
        x = [ioc[c] for c in text] # this is not a vector of count, but a sequence of indices
        
        #print(x, y)
        # get the scores
        scores = model(Tensor([x]).long())
        # take the argmax
        yhat = scores.argmax().item()
        #print(lng, text[r:r+k], scores.argmax(), y, sep='\t')
        # do they match or not ?
        confusion[y, yhat] += 1

    print('RNN', epoch)
    print_confusion(confusion, lngs)
    print()
