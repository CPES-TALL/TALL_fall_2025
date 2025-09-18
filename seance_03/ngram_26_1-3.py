# ---------------------------------------- -*- coding: utf-8 -*-
# N.B. Dans le dictionnaire le contexte est un tuple (même pour n=2)

import nltk
import random

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

def prediction_greedy(modele, n, contexte):
    '''prédit le mot qui doit suivre en prenant
    les derniers n-1 mots du contexte (si possible)'''
    if len(contexte) >= n-1:
        cle = tuple(contexte[-(n-1):])
        if cle in modele.keys():
            choix = modele[cle]
            return max(choix, key=choix.get)
    return 'XXXX'

def exactitude_corpus(corpus,modele,N):
    '''Tire Y=50 phrases au hasard dans le corpus et compare les
    prédictions du modèles avec les données gold pour tous les
    N-grammes de ces phrases, et renvoie le nombre de tirages et le
    nombre de hits'''
    Y = 100
    hits = 0
    nbpred = 0
    jeu_phrases_test = random.sample(corpus,Y)
    for phrase in jeu_phrases_test:
        for i in range(len(phrase)-(N-1)):
            nbpred += 1
            if prediction_greedy(modele, N, tuple(phrase[i:i+N-1])) == phrase[i+N-1]:
                hits += 1
    return hits, nbpred

def mesure_exactitudes(modele,N):
    for corpus_name in ["Candide.txt", "Oeuvres_Voltaire.txt", "Voyage_Ballon.txt"]:
        print("Avec des n-grammes du corpus %s, " % corpus_name, end="")
        hits, nbpred = exactitude_corpus(charge_corpus(corpus_name),modele,N)
        print("%d hits sur %d prédictions: accuracy = %.2f%%" % (hits, nbpred, 100*float(hits)/nbpred))

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

    mesure_exactitudes(modele,N)
