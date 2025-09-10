from tqdm import tqdm
from argparse import ArgumentParser
from pickle import load


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
    NotImplemented



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



# rules with previous and next words
# on va modifier forms pour y ajouter un token de début et de fin de phrase
forms = [['$$$'] + sen + ['$$$'] for sen in forms]

# on reprédit
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
    for i in range(1, len(sen)-1):
        f = sen[i]
        
        POS = 'UNKNOWN'
        for suf, pos in rules: # on teste nos règles de suffixe tour à tour quand une marche, on l'applique et on s'arrête pour le mot actuel
            if f.endswith(suf):
                POS = pos
                break

        pred[-1].append(POS)

mesure(gold, pred)
