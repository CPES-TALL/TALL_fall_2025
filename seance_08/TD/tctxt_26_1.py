# ---------------------------------------- -*- coding: utf-8 -*---------
# Fichier  : tctxt_26_1.py
# Contient : TP sur l'influence de la taille du contexte pour la
#            construction d'une matrice terme-terme   
#            Evaluation: k plus proches voisins de quelques termes
# Dépend   : nltk pour la tokenisation
#            re pour la version alt de tokenisation
# ----------------------------------------------------------------------
import math

# ----------------------------------------------------------------------
# Fonction : charge_corpus()
# Prototype: entree: nom de fichier (str)
#            sortie: corpus segmenté & tokenisé (liste de liste de str)
# Contenu  : ouvre le fichier txt dont le nom est donné en paramètre,
#            le découpe en phrases et en tokens en utilisant nltk
# --
# charge_corpus_alt() fait la même chose sans utiliser nltk, mais avec
#       re.split(), et un traitement minimal des apostrophe
# charge_corpus_spacy() fait la même chose en utilisant spacy, ce qui
#       est plus lourd mais linguistiquement bien mieux motivé. 
# ----------------------------------------------------------------------
from nltk import tokenize
# Si nécessaire, télécharger punkt_tab
# nltk.download('punkt_tab')

def charge_corpus(filename):
    corpus = []
    with open(filename, "r") as f:
        texte = f.read().strip().lower()
    for p in tokenize.sent_tokenize(texte):
        corpus.append(tokenize.word_tokenize(p))
    return corpus

import re

# def charge_corpus_alt(filename):
#     corpus = []
#     with open(filename, "r") as f:
#         texte = f.read().strip()
#     texte = texte.replace("'", "' ").lower()
#     texte = texte.replace('\n', ' ')
#     texte = re.split("[!?.;]", texte)
#     for s in texte:
#         corpus.append(s.split())
#     return corpus

# import spacy

# def charge_corpus_spacy(filename):
#     corpus = []
#     with open(filename, "r") as f:
#         texte = f.read()
#     nlp = spacy.load("fr_core_news_sm")
#     doc = nlp(texte)
#     for phrase in doc.sents:
#         tokens = [token.text for token in phrase]
#         corpus.append(tokens)
#     return corpus

# ----------------------------------------------------------------------
# Fonction : cree_vocabulaire()
# Prototype: entree: corpus (list(list(str)))
#            sortie: ensemble des types du corpus (set(str)
# Contenu  : construit la liste complète des tokens uniques (=types)
#            du corpus. Le vocabulaire est renvoyé trié par ordre
#            alphabétique.
# ----------------------------------------------------------------------
def cree_vocabulaire(corpus):
    vocab = set()
    for phrase in corpus:
        vocab.update(phrase)
    return sorted(vocab)

# ----------------------------------------------------------------------
# Fonction : cree_index()
# Prototype: entree: vocabulaire (list(str))
#            sortie: index : dict { mot: indice }
# Contenu  : produit un dictionnaire assocciant un entier à chaque mot
#            du vocabulaire
# --
# On peut connaître l'indice d'un mot m avec l'index créé: index[m]
# On peut le mot correspondant à un indice donné i en utilisant le
# vocabulaire : vocabulaire[i]
# ----------------------------------------------------------------------
def cree_index(vocab):
    return {mot: i for i, mot in enumerate(vocabulaire)}

# ----------------------------------------------------------------------
# Fonction : cree_matrice_terme_terme()
# Prototype: entree: corpus (list(list(str)))
#                    index (dict type: int)
#                    k = taille de la demi-fenêtre
#            sortie: demi matrice terme-terme
# Contenu  : Construit une matrice terme-terme en prenant pour chaque
#            mot (cible) un cbow de 1/2-taille k, en allant dans
#            l'index trouver les indices des mots correspondants, et
#            en incrémentant la demi-matrice initialisée à 0.
# --
# Comme c'est une demi-matrice, toute les cellules (x,y) ont des
# coordonnées telles que x > y (diagonale non remplie)
# --
# La fonction "explore_matrice()" compte le nombre de non-zéros de la
# matrice pour permettre une vérification sommaire
# ----------------------------------------------------------------------
def cree_matrice_terme_terme(corpus, index, k):
    n = len(index)
    matrice = [[0] * i for i in range(n)]

    for p in corpus:
        for i in range(len(p)):
            target = p[i]
            cbow = p[i-k:i] + p[i+1:i+k]
            i_t = index[target]
            for w in cbow:
                i_w = index[w]
                if i_t > i_w:
                    matrice[i_t][i_w] += 1
                elif i_w > i_t:
                    matrice[i_w][i_t] += 1
    return matrice

def explore_matrice(matrice):
    c = 0
    for i in range(len(matrice)):
        for j in range(len(matrice[i])):
            if matrice[i][j] != 0:
                c += 1
    return c
    # print("La matrice contient %d valeurs =/= de 0" % c)

# ----------------------------------------------------------------------
# Fonction : embedding()
# Prototype: entree: mot (str)
#                    index (dict type: int)
#                    matrice (list[list[int]])
#            sortie: vecteur (list[int])
# Contenu  : Crée un vecteur de dimention (len(index)) [ie taille du
#            vocabulaire] en récupérant les valeurs de la
#            demi-matrice.
# -- 
# Comme c'est une demi-matrice, toute les cellules (x,y) ont des
# coordonnées telles que x > y (diagonale non remplie)
# En principe le vecteur pour un mot d'indice k comprend les cellules:
# [k,0], [k,1], ... [k,k-1] 0,0 [k+1,k], [k+2,k], [k+3,k], ... [n-1,k]
# ----------------------------------------------------------------------
def embedding(mot, index, matrice):
    k = index[mot]
    vecteur = []
    for i in range(k):
        vecteur.append(matrice[k][i])
    vecteur.append(0)
    for i in range(k+1,len(index)):
        vecteur.append(matrice[i][k])
    return vecteur

# ----------------------------------------------------------------------
# Fonction : similarite_cosinus()
# Prototype: entree: vecteur1, vecteur2 (listes d'int)
#            sortie: réel entre 0 et 1
# Contenu  : Calcul classique du cosinus comme quotient du produit
#            scalaire sur le produit des normes
# ----------------------------------------------------------------------
def similarite_cosinus(vec1, vec2):
    produit_scalaire = sum(a * b for a, b in zip(vec1, vec2))
    norme1 = math.sqrt(sum(a * a for a in vec1))
    norme2 = math.sqrt(sum(b * b for b in vec2))
    if norme1 == 0 or norme2 == 0:
        return 0.0
    return produit_scalaire / (norme1 * norme2)

# ----------------------------------------------------------------------
# Fonction : dix_voisins()
# Prototype: entree: mot (str)
#                    index (dict { str : int }
#                    vocab (list[str])
#                    matrice terme-terme list[list[int]]
#            sortie: /
#            effet de bord: affichage des 10 mots les plus proches
# Contenu  : à partir de l'embedding du mot obtenu dans la matrice,
#            calcule les similarités cosinus avec tout le reste du
#            vocabulaire, et affiche les 10 mots les plus proches
# ----------------------------------------------------------------------
def dix_voisins(mot,index,vocab,matrice,k):
    v1 = embedding(mot,index,matrice)
    similarites = []
    for m in vocab:
        similarites.append(similarite_cosinus(v1,embedding(m,index,matrice)))

    scores = []
    for i,val in enumerate(similarites):
        scores.append((val, i))

    scores = sorted(scores)

    for i, (val, idx) in enumerate(scores[-10:]):
        print("k = %d, cible = %-12s| %2d ieme mot le plus proche: %12s (%.2f)"
              % (k,mot,10-i,vocabulaire[idx], val))
    print()

# **********************************************************************
# VARIABLES GLOBALES & PARAMÈTRES

# Corpus de travail
corpus_name = "Candide.txt"

# ----------------------------------------------------------------------
# PROGRAMME PRINCIPAL

# Chargement du corpus
print("Chargement du corpus... ", end='')
corpus = charge_corpus(corpus_name)
print("%d phrases chargées. " % len(corpus))
# Création du vocabulaire (liste des types)
vocabulaire = cree_vocabulaire(corpus)
print("le vocabulaire comprend %d mots." % len(vocabulaire))
# Création d'un index (associant un nombre à chaque type)
index = cree_index(vocabulaire)
# Création d'une matrice terme-terme pour une demi-fenêtre k
k = 5
print("Création d'une matrice terme-terme...")
matrice = cree_matrice_terme_terme(corpus, index, k)
print("La matrice créée contient %d valeurs non nulles." % explore_matrice(matrice))
# Vérification sommaire que la matrice n'est pas vide
# explore_matrice(matrice)
# Récupération et affichage des 10 mots les plus proches d'un mot cible
w = "soldat"
print("Recherche des dix plus proches voisins du mot %s..." % w)
dix_voisins(w,index,vocabulaire,matrice,k)

# Boucle de test sur un ensemble de mots et un ensemble de valeurs pour k
# (à dé-commenter quand tout le reste a été testé)

# # Liste de mots pour tester
# lmots = ['candide', 'est', 'un', 'fameux', 'soldat', 'château', 'grand', 'pour', 'dix', 'lui']
# # Liste de valeurs de k pour tester
# valk = [1,3,5,7,12,15]

# for k in valk[:2]:
#     matrice = cree_matrice_terme_terme(corpus, index, k)
#     for w in lmots:
#         dix_voisins(w,index,vocabulaire,matrice,k)
