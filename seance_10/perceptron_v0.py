# ---------------------------------------- -*- coding: utf-8 -*---------
# Fichier  : perceptron_ng0.py
# Contient : TP sur le perceptron, version non graphique
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Fonction : prediction()
# Prototype: entrée: x_1 et x_2
#            poids: triplets de poids
#            sortie: valeur prédite par le perceptron défini par ces poids
# Contient : applique le calcul de prédiction
#            (combinaison linéaire poids x entrée + fonction de transfert)
# ----------------------------------------------------------------------
def prediction(x1, x2, poids):
    somme = poids[0] + x1 * poids[1] + x2 * poids[2]
    activ = 0 if somme <= 0 else 1
    return activ

# ----------------------------------------------------------------------
# Fonction : epoque()
# Prototype: entrée: ref (liste de triplets d'apprentissage)
#            poids: triplets de poids modifiés par effet de bord
#            sortie: /
# Contient : boucle sur toutes les données de la référence,
#            calcule l'erreur et modifie les poids passés en paramètre
# --
# La version _g calcule en plus le score global d'erreur (somme des
# valeurs absolues des erreurs), et renvoie cette valeur
# ----------------------------------------------------------------------
def epoque(ref, poids, nu):
    for x1, x2, y in ref:
        erreur = y - prediction(x1, x2, poids)
        if erreur:
            poids[0] += nu * erreur
            poids[1] += nu * erreur * x1
            poids[2] += nu * erreur * x2

# ----------------------------------------------------------------------
# Fonction : erreur_globale()
# Prototype: entrée: ref (liste de triplets d'apprentissage)
#            poids: triplets de poids modifiés par effet de bord
#            sortie: erreur globale pour un ensemble de prédictions
# Contient : boucle sur toutes les données de la référence,
#            cumule les valeurs absolues d'erreur pour chaque prédiction
# ----------------------------------------------------------------------
def erreur_globale(ref, poids):
    erreur_globale = 0
    for x1, x2, y in ref:
        erreur = y - prediction(x1, x2, poids)
        erreur_globale += abs(erreur)
    return erreur_globale

# ----------------------------------------------------------------------
# Fonction : apprentissage_simple()
# Prototype: entrée: ref (liste de triplets d'apprentissage)
#            sortie: /
#            effet de bord: affichage résultats apprentissage
# Contient : nombre d'époques prédéfinie, affichage erreur globale
#      nmax: nombre d'époques maximales prédéfinie, stop si convergence
#      conv: arrêt de l'apprentissage dès qu'il y a convergence
#            (avec une borne (1000) sur le nombre d'époques)
# ----------------------------------------------------------------------
def apprentissage_simple(ref,n=5):
    poids = init_poids()
    print(poids)
    for i in range(n):
        epoque(ref,poids,nu)
    print("Après %d époques, erreur globale: %d" % (n, erreur_globale(ref,poids)))
    print(poids)

# **********************************************************************
# VARIABLES GLOBALES & PARAMÈTRES
# Données de référence: fonction qu'on cherche à approximer
# (x1, x2, y)
#      x1, x2 \in {0, 1}
#      y (classe) \in {0, 1}
AND = [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
data_ref = AND

# Taux d'apprentissage
nu = .1

# Les poids qui représentent les propriétés du neurone
def init_poids():
    return [-1, 1.4, -1] 

# **********************************************************************
# PROGRAMME PRINCIPAL
    
apprentissage_simple(data_ref, 15)

