#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:51:19 2022

@author: praneeth44
"""

from math import log
import numpy as np
import json

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence

    max_T, max_A = data.shape

    # Loop over time
    for t in range(max_T):
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            # Loop over possible alphabet outputs
            for c in range(max_A - 1):
                candidate = [seq + [c], score - log(data[t, c])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


    
with open('seq_collapse.txt') as json_file:
    new_data = json.load(json_file)
    
result = beam_search_decoder(np.array(new_data['logits']), 100)


final=[]
for j in (result):
    ans=[]
    for i in enumerate(j[0]):
        ans.append(i[1])
    final.append(ans.copy())
print(final)


import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

position = new_data['alphabet']
print('possss',len(position))
chars = [i for i in range(50)]

harvest = np.array(final)


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(position)), labels=position)
ax.set_yticks(np.arange(len(chars)), labels=chars)

plt.rcParams["figure.figsize"] = (5,5)
plt.savefig('seq2.png')
plt.show()
    
