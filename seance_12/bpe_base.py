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

# for as long as we have no reach the target
tokens = []
mergers = []
it = 0
while len(tokens) < target - len(characters):
    
    # first let's initialize our bigram count dict
    # and also count the occurence of each token : note that by merging token, an old token could disappear

    ()
