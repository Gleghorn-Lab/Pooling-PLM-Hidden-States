import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

class SubcellularLocalizationDataset(Dataset):
    def __init__(self, dataframe, embedding_dict):
        self.dataframe = dataframe
        self.embedding_dict = embedding_dict
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        acc = self.dataframe.iloc[idx]['ACC']
        if acc in self.embedding_dict:
            embedding = self.embedding_dict[acc]
        else:
            #raise Exception(f"The following protein doesn't seem to have pooled embeddings generated {acc}")
            print(f"The following protein doesn't seem to have pooled embeddings generated {acc}")
            return None, None
        labels = self.dataframe.iloc[idx][1:].to_numpy().astype(float)  # Assuming labels start from 3rd column
        #labels = self.dataframe.iloc[idx][2:].to_numpy().astype(float)  # Assuming labels start from 3rd column
        labels = torch.tensor(labels, dtype=torch.float32)
        return embedding, labels