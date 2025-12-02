"""
CPES SDAC 2025-2026
TAL²
Mathieu Dehouck

Tokenization in the age of LLMs
"""

from tqdm import tqdm
from collections import defaultdict
from time import time

# first load some text
data = []
path = 'data/en'
path = 'data/en_Cather_the_troll_garden'
for l in open(path):
    l = l.strip()
    if l != '':
        data.append(l)


# make a pre-tokenization on spaces
# in practice we also do it for punctuation but for simplicity we'll ignore it now
# we need to make the beginning of a word, we'll use a special ^ character for that
data = [['^' + w for w in l.split()] for l in data]
tokenized = [[[c for c in w] for w in l] for l in data]

words = defaultdict(int)
characters = set()
# now get all the words and all the characters
for l in tokenized:  # for each data line
    for w in l: # for each pretokenized word
        characters.update(w) # update the character set
        words[tuple(w)] += 1        # increment the count
        
print('#of characters :', len(characters), '#of words :', len(words), 'length of document :', sum([len(w)*v for w,v in words.items()]), sep='\t')


# let's choose the size of the target vocabulary
target = 500

# we will merge tokens two by two in such a way as to reduce the length of the document as much as possible, given the desired target
# this means that we need to choose greedily the best merger at each step

t0 = time()

# for as long as we have no reach the target
tokens = []
mergers = []
it = 0
while len(tokens) < target - len(characters):
    
    # first let's initialize our bigram count dict
    # and also count the occurence of each token : note that by merging token, an old token could disappear
    bigrams = defaultdict(int)
    tok_count = defaultdict(int)
    for w, cnt in tqdm(words.items(), leave=False):
        for i in range(len(w)-1):
            bigrams[w[i:i+2]] += cnt

        for t in w:
            tok_count[t] += cnt

    safe_tokens = [t for t in tokens if tok_count[t] != 0]
    if safe_tokens != tokens:
        print(">>>>", [t for t in tokens if t not in safe_tokens])
    tokens = safe_tokens
    
    # let's find the most frequent bigram
    sorted_bis = sorted(bigrams.items(), key=lambda x: x[1])
    bigram, occ = sorted_bis[-1]
    print(bigram, occ, sorted_bis[-5:], sep='\t')

    tokens.append(''.join(bigram)) # add a new token in the vocabulary
    mergers.append(bigram)
    
    # we need to update the forms containing the bigram, replacing it with its own new token
    new_words = defaultdict(int)
    for w, cnt in words.items():
        i = 0
        nw = []
        while i < len(w):
            #print(w[i:i+2], bigram)
            if w[i:i+2] == bigram:
                nw.append(''.join(bigram))
                i += 2
            else:
                nw.append(w[i])
                i += 1

        #print(nw)
        new_words[tuple(nw)] = cnt

    words = new_words

    print(it, '#of tokens :', len(tokens), 'length of document :', sum([len(w)*v for w,v in words.items()]), sep='\t')
    it += 1
    #print(tokens)

t1 = time()
print(t1 - t0)

print()

tok_count = defaultdict(int)
for w, cnt in tqdm(words.items(), leave=False):
    #print(w, cnt)
    for t in w:
        tok_count[t] += cnt

print()
for t in tokens:
    print(t, tok_count[t], sep='\t')


# now we print the tokenized text, little by little so that you see the process unfold :

print()
for l in tqdm(tokenized):
    for big in mergers:
        big = list(big)

        nl = []
        for w in l:
            i = 0
            nw = []
            while i < len(w):
                if w[i:i+2] == big:
                    nw.append(''.join(big))
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1

            nl.append(nw)

        l = nl

    print(' '.join(' '.join(nw) for nw in nl))
