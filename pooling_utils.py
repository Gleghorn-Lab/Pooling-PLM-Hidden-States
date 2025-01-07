import torch
import numpy as np
from typing import Tuple


class Pooler:
    def __init__(self, hf_dataset, emb_dict):
        self.train_seqs, self.train_labels = hf_dataset['train']['seqs'], hf_dataset['train']['labels']
        self.valid_seqs, self.valid_labels = hf_dataset['valid']['seqs'], hf_dataset['valid']['labels']
        self.test_seqs, self.test_labels = hf_dataset['test']['seqs'], hf_dataset['test']['labels']
        self.emb_dict = emb_dict

        self.pooling_options = {
            'mean': self.mean_pooling,
            'max': self.max_pooling,
            'min': self.min_pooling,
            'norm': self.norm_pooling,
            'prod': self.prod_pooling,
            'median': self.median_pooling,
            'std': self.std_pooling,
            'var': self.var_pooling,
            'cls': self.cls_pooling,
            'eos': self.eos_pooling
        }

    def mean_pooling(self, emb): # (L, d) -> (d)
        return emb.mean(dim=0)

    def max_pooling(self, emb): # (L, d) -> (d)
        return emb.max(dim=0).values
    
    def min_pooling(self, emb): # (L, d) -> (d)
        return emb.min(dim=0).values

    def norm_pooling(self, emb): # (L, d) -> (d)
        return emb.norm(dim=0, p=2)

    def prod_pooling(self, emb): # (L, d) -> (d)
        length = emb.shape[0]
        return emb.prod(dim=0) / length

    def median_pooling(self, emb): # (L, d) -> (d)
        return emb.median(dim=0).values
    
    def std_pooling(self, emb): # (L, d) -> (d)
        return emb.std(dim=0)
    
    def var_pooling(self, emb): # (L, d) -> (d)
        return emb.var(dim=0)

    def cls_pooling(self, emb): # (L, d) -> (d)
        return emb[0, :]
    
    def eos_pooling(self, emb): # (L, d) -> (d)
        return emb[-1, :]

    def get_pooled_embeddings(self, pooling_types: Tuple[str, str]): # [mean, max]
        """
        pooler = Pooler(hf_dataset, emb_dict)
        pooling_types = ['mean', 'max']
        X_train, y_train, X_valid, y_valid, X_test, y_test = pooler.get_pooled_embeddings(pooling_types)
        # X (n_samples, n_features)
        # y (n_samples)
        """
        for pool_type in pooling_types:
            if pool_type not in self.pooling_options:
                raise ValueError(f"Invalid pooling type: {pool_type}")
        
        train_embeddings, valid_embeddings, test_embeddings = [], [], []
        train_labels = np.array(self.train_labels)
        valid_labels = np.array(self.valid_labels)
        test_labels = np.array(self.test_labels)

        # concatenate all pooling for each sequence
        for seq in self.train_seqs:
            final_emb = []
            emb = self.emb_dict[seq] # (L, d)
            for pool_type in pooling_types:
                pooled_emb = self.pooling_options[pool_type](emb) # (d)
                final_emb.append(pooled_emb)
            final_emb = torch.cat(final_emb, dim=0).flatten() # (d * len(pooling_type))
            train_embeddings.append(final_emb)

        train_embeddings = torch.stack(train_embeddings).numpy().astype(np.float32)

        for seq in self.valid_seqs:
            final_emb = []
            emb = self.emb_dict[seq]
            for pool_type in pooling_types:
                pooled_emb = self.pooling_options[pool_type](emb)
                final_emb.append(pooled_emb)
            final_emb = torch.cat(final_emb, dim=0).flatten()
            valid_embeddings.append(final_emb)

        valid_embeddings = torch.stack(valid_embeddings).numpy().astype(np.float32)

        for seq in self.test_seqs:
            final_emb = []
            emb = self.emb_dict[seq]
            for pool_type in pooling_types:
                pooled_emb = self.pooling_options[pool_type](emb)
                final_emb.append(pooled_emb)
            final_emb = torch.cat(final_emb, dim=0).flatten()
            test_embeddings.append(final_emb)

        test_embeddings = torch.stack(test_embeddings).numpy().astype(np.float32)

        return train_embeddings, train_labels, valid_embeddings, valid_labels, test_embeddings, test_labels
