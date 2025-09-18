# ---------------------------------------- -*- coding: utf-8 -*-
# N.B. Dans le dictionnaire le contexte est un tuple (même pour n=2)

import nltk

# Si nécessaire, télécharger punkt_tab
#nltk.download('punkt_tab')

def charge_corpus(filename):
    ''' retourne une liste de phrases
    qui sont elles-même des listes de tokens'''
    corpus = []
    with open(filename, "r") as f:
        for p in nltk.tokenize.sent_tokenize(f.read()):
            corpus.append(nltk.word_tokenize(p))
    return corpus

def construit_modele(corpus, n=2):
    modele = {}
    for phrase in corpus:
        phrase = ['<s>'] + phrase + ['</s>']
        for i in range(len(phrase) - n + 1):
            contexte, mot = tuple(phrase[i:i+n-1]), phrase[i+n-1]
            if contexte in modele.keys():
                if mot in modele[contexte].keys():
                    modele[contexte][mot] += 1
                else:
                    modele[contexte][mot] = 1
            else:
                modele[contexte] = { mot : 1 }
    return modele

def verification_modele(m):
    cle = ('<s>',)
    print(modele[cle])
    cle = ('petite',)
    print(modele[cle])
    

# VARIABLES GLOBALES & PARAMÈTRES

# Corpus de travail
corpus_name = "Candide.txt"

# Taille des n-grammes
N = 2


# PROGRAMME PRINCIPAL
if __name__ == '__main__':
    print("Chargement du corpus... ", end='')
    corpus = charge_corpus(corpus_name)
    print("%d tokens chargés. " % len(corpus))
    modele = construit_modele(corpus, N)
    print("Modèle n-gramme calculé, n=%d" % N)

#    verification_modele(modele)
