import itertools
import os
from collections import Counter

def parser_cnt(path):
    with open(path, 'r') as file:
        data = file.readlines()
    data = [value.split('\t')[1].strip() for value in data]
    data = [list(value) for value in list(set(data))]
    list_chars = itertools.chain(*data)
    result = dict(Counter(list_chars))
    sum_value = sum([1/value for value in result.values()])
    for key in result:
        result[key] = (1 / result[key]) / sum_value
    return result

class Vocab():
    def __init__(self, chars, path_train: str = None):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c:i+4 for i, c in enumerate(chars)}

        self.i2c = {i+4:c for i, c in enumerate(chars)}
        if path_train is not None:
            weight = parser_cnt(path_train)
            print(weight)
            
            self.weight_contribution = [weight[character] for character in chars]
            self.weight_contribution.inset(0, 0)
            self.weight_contribution.inset(0, max(self.weight_contribution))
            self.weight_contribution.inset(0, max(self.weight_contribution))
            self.weight_contribution.inset(0, 0)
            
            print(weight)
        else:
            self.weight_contribution = None
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]
    
    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent
    
    def __len__(self):
        return len(self.c2i) + 4
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars
