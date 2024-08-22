# -*- coding: utf-8 -*-
"""
@author: avasque1@jh.edu
"""
import torch
from torch.utils.data import Dataset

from utils import *

class SMILESDataset(Dataset):
    '''
    Custom Dataset for SMILES Strings
    '''
    def __init__(self, smiles_list: list, vocab: dict, max_length: int) -> None:
        '''
        Custom smiles dataset
        '''
        self.smiles_list = smiles_list
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        '''
        Required
        '''
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> torch.tensor:
        '''
        Required
        '''
        smiles = self.smiles_list[idx]
        encoded = encode_smiles(smiles, self.vocab)
        padded = encoded + [0] * (self.max_length - len(encoded))
        return torch.tensor(padded, dtype=torch.long)