# ---------------------------------------- -*- coding: utf-8 -*---------
# Fichier  : analogie_26_2.py
# Contient : TP sur le raisonnement analogique dans les embeddings
#            
# Dépend   : re pour la  tokenisation
#            math pour sqrt()
# --
# v2: corrigé TP octobre 2025
# ----------------------------------------------------------------------
import math

# ----------------------------------------------------------------------
# Fonction : charge_corpus()
# Prototype: entree: nom de fichier (str)
#            sortie: corpus segmenté & tokenisé (liste de liste de str)
# Contenu  : ouvre le fichier txt dont le nom est donné en paramètre,
#            le découpe en phrases et en tokens de façon très grossière
#            en utilisant la bibliothèque "re"
# --
# 
# ----------------------------------------------------------------------
import re

def charge_corpus(filename):
    corpus = []
    with open(filename, "r") as f:
        texte = f.read().strip()
    texte = texte.replace("'", "' ").lower()
    texte = texte.replace('\n', ' ')
    texte = re.split("[!?.;]", texte)
    for s in texte:
        corpus.append(s.split())
    return corpus

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
# Fonction : embedding_indice()
# Prototype: entree: indice (int)
#                    matrice (list[list[int]])
#            sortie: vecteur (list[int])
# Contenu  : Crée un vecteur de dimension (len(index)) [ie taille du
#            vocabulaire] en récupérant les valeurs de la
#            demi-matrice pour l'indice passé en paramètre. 
# -- 
# Comme c'est une demi-matrice, toute les cellules (x,y) ont des
# coordonnées telles que x > y (diagonale non remplie)
# En principe le vecteur pour un mot d'indice k comprend les cellules:
# [k,0], [k,1], ... [k,k-1] 0,0 [k+1,k], [k+2,k], [k+3,k], ... [n-1,k]
# ----------------------------------------------------------------------
def embedding_indice(indice, matrice):
    k = indice
    vecteur = []
    for i in range(k):
        vecteur.append(matrice[k][i])
    vecteur.append(0)
    for i in range(k+1,len(index)):
        vecteur.append(matrice[i][k])
    return vecteur

# ----------------------------------------------------------------------
# Fonction : embedding()
# Prototype: entree: mot (str)
#                    index (dict type: int)
#                    matrice (list[list[int]])
#            sortie: vecteur (list[int])
# Contenu  : renvoie l'embedding correspondant à un mot en récupérant
#            l'indice du mot (via l'index) et en appelant la fonction
#            embedding_indice() qui donne le vecteur à partir de la
#            matrice
# ----------------------------------------------------------------------
def embedding(mot, index, matrice):
    k = index[mot]
    return embedding_indice(k, matrice)

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
# Fonction : somme_vecteurs()
# Prototype: entree: vecteur1, vecteur2 (listes d'int)
#            sortie: vecteur somme
# Contenu  : Calcul classique coordonnée par coordonnée
# ----------------------------------------------------------------------
def somme_vecteurs(vec1, vec2):
    s = []
    for x1, x2 in zip(vec1, vec2):
        s.append(x1+x2)
    return s

# ----------------------------------------------------------------------
# Fonction : moins_vecteur()
# Prototype: entree: vecteur (listes d'int)
#            sortie: vecteur précédent * -1
# Contenu  : Calcul classique coordonnée par coordonnée
# ----------------------------------------------------------------------
def moins_vecteur(v):
    s = []
    for x in v:
        s.append(-x)
    return s

# ----------------------------------------------------------------------
# Fonction : n_voisins_vect()
# Prototype: entree: vecteur
#                    matrice terme-terme list[list[int]]
#                    n (int)
#            sortie: liste des index des n embeddings les plus proches
#                    du vecteur donné en entrée
#            effet de bord: /
# --
# Inspiré de la fonction "dix_voisins()" du TP sur la taille du contexte
# mais cette fonction ne manipule que des embeddings, et elle renvoie
# les indices des n mots les plus proches. 
# ----------------------------------------------------------------------
def n_voisins_vect(v,matrice,n=10):
    similarites = []
    for i in range(len(matrice)): 
        similarites.append(similarite_cosinus(v,embedding_indice(i, matrice)))

    scores = []
    for i,val in enumerate(similarites):
        scores.append((val, i))

    scores = sorted(scores)
    return scores[-n:]

def affichage_voisins(liste):
    n = len(liste)
    for i, (val, idx) in enumerate(liste):
        print("%2d ieme mot le plus proche du point: %12s (%.2f)"
              % (n-i,vocabulaire[idx], val))
    print()

# ----------------------------------------------------------------------
# Fonction : affiche_analogie()
# Prototype: entrée: 4 mots dans le bon ordre
#                    index
#                    matrice
#                    n (int) : nombre de voisins (défaut 10)
#            sortie: /
#            effet de bord: affichages
# Contient : récupère les vecteurs des 4 mots, pour lesquels l'analogie
#            visée est m[0] - m[1] + m[2] =?= m[3]
#            calcule le vecteur y = v0 - v1 + v2
#            calcul la similarité de y avec tous les mots du vocabulaire
#            et affiche les n plus proches;
#            (re-)calcule aussi la similarité de y avec m[3] et l'affiche
# ----------------------------------------------------------------------
def affiche_analogie(mots,index,matrice,n=10):
    # Récupération des 4 vecteurs
    vect = []
    for m in mots:
        vect.append(embedding(m,index,matrice))
    # Calcul de l'embedding de vect0 - vect1 + vect3
    print("Calcul du vecteur v(%s) - v(%s) + v(%s)..." % (mots[0],mots[1],mots[2]))
    y = somme_vecteurs(somme_vecteurs(vect[0],moins_vecteur(vect[1])),vect[2])
    print("Recherche des %d mots les plus proches du point trouvé..." % n)
    affichage_voisins(n_voisins_vect(y,matrice,n))
    # Le mot correct est connu: affichage de sa distance
    print("Similarité entre le point trouvé et le mot %s: %.2f" % (mots[3],similarite_cosinus(y,vect[3])))

# **********************************************************************
# VARIABLES GLOBALES & PARAMÈTRES

# Corpus de travail
corpus_name = "Candide.txt"

# Préparation pour le calcul des 4 équations analogiques possibles pour
# un quadruplet de mots.
# On code un quadruplet comme une matrice 2 x 2
quadruplets = [
#    [('père','mère'),('fils','fille')],
    [('le','la'),('un','une')],
#    [('candide','cunégonde'),('homme','femme')],
]
# Les équations analogiques sont données ci-dessous en supposant que les
# mots sont dans une matrice 2x2. 
equations = [
    [(0,0),(1,0),(1,1),(0,1)],
    [(0,0),(0,1),(1,1),(1,0)],
    [(1,0),(0,0),(0,1),(1,1)],
    [(1,0),(1,1),(0,1),(0,0)]
]
# Avec le tableau        0       1  
#                    | père  | mère  |  0
#                    -----------------
#                    | fils  | fille |  1
# La première équation correspondra à :
# (0,0)- (1,0)+ (1,1) =?= (0,1)
# père - mère + fille =?= fils

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
k = 6
print("Création d'une matrice terme-terme (taille de le fenêtre: %d)..." % k)
matrice = cree_matrice_terme_terme(corpus, index, k)
print("La matrice créée contient %d valeurs non nulles." % explore_matrice(matrice))

# Petit programme annexe pour lancer les tests analogiques pour un quadruplet
def boucle_quadruplet_analogique(mots,index,matrice):
    for eq in equations:
        m = []
        for i in [0,1,2,3]:
            (a,b) = eq[i]
            m.append(mots[a][b])
        affiche_analogie(m,index,matrice,5)

for q in quadruplets:
    boucle_quadruplet_analogique(q,index,matrice)


