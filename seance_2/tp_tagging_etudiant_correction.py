from tqdm import tqdm
from argparse import ArgumentParser
from pickle import load




"""

pour faire tourner ce code avec VSCode : 

- lancez VSCode;
- fichier/file > open folder/ouvrir un dossier;
- choisir le dossier contenant ce script ainsi que les fichier pkl et conllu.

- clickez sur la flèche : Run Python File
- normalement vous aurez une erreur, un fichier dev manquant !   si ce n'est pas le cas, vous n'êtes pas au bon endroit. fermer cette fenêtre et recommencez.

- pour l'erreur de dev : placez vous maintenant dans la console, appuyer sur la flèche vers les haut de votre clavier, cela rapelle la commande lancée précédemment.
- au bout de la commande qui ressemble à ".../python3... .../tp_tagging_etudiant_correction.py" ajouter après une espace fro_profiterole-ud-dev.conllu puis tapez enter/entrée.
- normalement ça marche.
- pour faciliter la saisie de noms de fichiers, vous pouvez utiliser l'autocomplétion avec tab (la touche au dessus de capslock, la touche deux fois au dessus du MAJuscule de gauche).

- si vous avez une erreur en lien avec tqdm, vous pouvez commenter la première ligne et remplacer toutes les instances de tqdm(xxx) par xxx.
- si tout marche mais que vous n'avez pas le résultat indiqué dans le pdf, c'est peut être un problème d'encodage, il faut peut être rajouter encoding="utf8" ou qqchose comme ça dans open(fname).

"""




# read args from the command line
ap = ArgumentParser()
ap.add_argument('dev', help='path to the dev file')
args = ap.parse_args()


# a function to read the data
def read_conllu(fname):
    data = []
    with open(fname) as f:
        for l in tqdm(f): # tqdm is a library that makes nice loading bars in the console
            l = l.strip() # remove the trailing spaces and line breaks
            if l == '' or l[0] == '#': # ignore empty and comment lines
                continue

            l = l.split('\t') # split on tabs, some tokens contain spaces, so it is important to split on tabs
            if '.' in l[0] or '-' in l[0]: # these represent mergers (du = de le) or ellipsis (je mange des pommes et toi ... des poires)
                continue

            if l[0] == '1': # a new sentence starts here, 1 is the index of the first true token a the sentence, we keep 0 for a dummy root
                data.append([])

            data[-1].append(l)

    return data # data is a list of list of list


# read the data
dev = read_conllu(args.dev)



def mesure(gold, pred):
    total = 0
    true = 0
    false = 0
    missing = 0

    for i in range(len(gold)):  # loop over all the sentences
        for j in range(len(gold[i])): # loop over all the token of the sentence
            if gold[i][j] == pred[i][j]: # same preidcted POS as gold
                true += 1
            elif pred[i][j] == 'UNKNOWN':
                false += 1
            else:
                missing += 1

            total += 1
            
    print(total, true, false, missing, sep='\t')
    print(total, true/total*100, false/total*100, missing/total*100, sep='\t')

    return true / total * 100



# example of dictionary based POS tagging

# prepare the data
gold = []
forms = []

for sen in dev:
    gold.append([]) # add a new list for the new sentence in both gold POS and forms
    forms.append([])

    for w in sen:
        gold[-1].append(w[3]) # append the gold POS to gold
        forms[-1].append(w[1]) # get a parallel list of forms (tokens)

        
form_to_pos = load(open('fro_dict.pkl', 'rb')) # this is a dictionary of old french POS I prepared

pred = [] # predictions
for sen in forms:
    pred.append([]) # add a new list for the current sentence
    for f in sen:
        pred[-1].append(form_to_pos.get(f, 'UNKNOWN')) # if f is not in the dictionary, we set a dummy UNKNOWN POS


mesure(gold, pred)




# simple rules
form_to_pos = load(open('fro_dict_small.pkl', 'rb')) # this is a small dictionary of old french POS I prepared, it contains only the words that appear more that 1% of the times in the train file
pred = []

rules = [('rent', 'VERB'),
         ('ient', 'VERB'),
         ('ment', 'ADV'),
         ('oi', 'PRON'),
         ('i', 'VERB'),
         ('or', 'NOUN'),
         ('r', 'VERB')]

for sen in forms:
    pred.append([]) # add a new list for the current sentence
    for f in sen:
        if f in form_to_pos: # si le mot est dans le petit dictionnaire des mots courants, on l'utilise
            pred[-1].append(form_to_pos[f])

        else: # sinon
            POS = 'UNKNOWN'
            for suf, pos in rules: # on teste nos règles de suffixe tour à tour quand une marche, on l'applique et on s'arrête pour le mot actuel
                if f.endswith(suf):
                    POS = pos
                    break

            pred[-1].append(POS)
        
mesure(gold, pred)




# if you don't even have a small dictionary
form_to_pos = None
pred = []

rules = [('rent', 'VERB'),
         ('ient', 'VERB'),
         ('ment', 'ADV'),
         ('oi', 'PRON'),
         ('i', 'VERB'),
         ('or', 'NOUN'),
         ('r', 'VERB')]

for sen in forms:
    pred.append([]) # add a new list for the current sentence
    for f in sen:
        POS = 'UNKNOWN'
        for suf, pos in rules: # on teste nos règles de suffixe tour à tour quand une marche, on l'applique et on s'arrête pour le mot actuel
            if f.endswith(suf):
                POS = pos
                break

        pred[-1].append(POS)

mesure(gold, pred)