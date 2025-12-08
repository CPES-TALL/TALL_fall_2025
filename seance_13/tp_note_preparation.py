  """
CPES DAC 2025-2026
TAL²
Mathieu Dehouck
"""

"""
Pour vous assurez que votre machine est prête pour le TP noté.
"""


from random import randint, seed
from os import scandir
import torch
from torch import Tensor, nn
from tqdm import tqdm, trange
import numpy as np

# loading a transformer model and the tokenizer that comes with it
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)


# set the seeds to be as reproducible as possible
seed(0)
torch.manual_seed(0)

txt = 'a big cat slap on the mat .'
toks_ids = tokenizer(txt, add_special_tokens=False)

embs = model(Tensor([toks_ids['input_ids']]).long()) # given a tensor of indices, the transformer return a series of representations most importantly, embs[0] and embs[1]
# 4 % what are the sizes/dimensions of embs[0] and embs[1], what do you infer from it ?

print(tuple(embs[0].shape) == (1, 8, 30522), len(embs[1]) == 13, tuple(embs[1][0].shape) == (1, 8, 768))

### si tout c'est bien passé vous devez avoir True True True et un message sur des poids non initialisés
