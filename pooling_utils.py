import torch
import numpy as np
from typing import Tuple
from itertools import combinations


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

    def mean_pooling(self, emb): # (L, d) -> (d,)
        return emb.mean(dim=0)

    def max_pooling(self, emb): # (L, d) -> (d,)
        return emb.max(dim=0).values
    
    def min_pooling(self, emb): # (L, d) -> (d,)
        return emb.min(dim=0).values

    def norm_pooling(self, emb): # (L, d) -> (d,)
        return emb.norm(dim=0, p=2)

    def prod_pooling(self, emb): # (L, d) -> (d,)
        length = emb.shape[0]
        return emb.prod(dim=0) / length

    def median_pooling(self, emb): # (L, d) -> (d,)
        return emb.median(dim=0).values
    
    def std_pooling(self, emb): # (L, d) -> (d,)
        return emb.std(dim=0)
    
    def var_pooling(self, emb): # (L, d) -> (d,)
        return emb.var(dim=0)

    def cls_pooling(self, emb): # (L, d) -> (d,)
        return emb[0, :]

    def eos_pooling(self, emb): # (L, d) -> (d,)
        return emb[-1, :]

    def pool_seqs(self, seqs, pooling_types: Tuple[str, str], modification: str = None):
        embeddings = []
        for seq in seqs:
            emb = self.emb_dict[seq]
            final_emb = []
            for pool_type in pooling_types:
                pooled_emb = self.pooling_options[pool_type](emb)
                final_emb.append(pooled_emb.reshape(1, -1)) # (1, e)
            final_emb = torch.cat(final_emb, dim=0) # (len(pooling_types), d)
            if modification == 'sum':
                final_emb = final_emb.sum(dim=0) # (d,)
            elif modification == 'product':
                final_emb = final_emb.prod(dim=0) # (d,)
            else:
                final_emb = final_emb.flatten() # (d * len(pooling_types),)
            embeddings.append(final_emb)
        return embeddings

    def get_pooled_embeddings(self, pooling_types: Tuple[str, str], modification: str = None): # [mean, max]
        """
        modification: sum or product
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
        train_embeddings = self.pool_seqs(self.train_seqs, pooling_types, modification)
        valid_embeddings = self.pool_seqs(self.valid_seqs, pooling_types, modification)
        test_embeddings = self.pool_seqs(self.test_seqs, pooling_types, modification)

        train_embeddings = torch.stack(train_embeddings).numpy().astype(np.float32)
        valid_embeddings = torch.stack(valid_embeddings).numpy().astype(np.float32)
        test_embeddings = torch.stack(test_embeddings).numpy().astype(np.float32)

        return train_embeddings, train_labels, valid_embeddings, valid_labels, test_embeddings, test_labels
    

if __name__ == '__main__':
    # py -m pooling_utils
    from transformers import AutoModel
    from datasets import load_dataset
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    model = AutoModel.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True, torch_dtype=torch.float16)
    def dataset_embed(dataset_name):
        data = load_dataset(dataset_name)
        all_seqs = data['train']['seqs'] + data['valid']['seqs'] + data['test']['seqs']
        print(f"Number of sequences: {len(all_seqs)}")
        emb_dict = model.embed_dataset(
            sequences=all_seqs, # list of protein strings
            batch_size=8, # embedding batch size
            max_len=2048, # truncate to max_len
            full_embeddings=True, # return residue-wise embeddings
            full_precision=False, # store as float32
            num_workers=0, # data loading num workers
            sql=False, # return dictionary of sequences and embeddings
            )
        return data, all_seqs, emb_dict

    pooling_options = ['mean', 'max', 'min', 'norm', 'median', 'std', 'var', 'cls', 'eos']
    modifications = [None, 'sum', 'product']

    # Generate experiments
    pairwise_combinations = list(combinations(pooling_options, 2))  # Combinations of two pooling methods
    product_methods = [(a, b) for a in pairwise_combinations for b in modifications if b == 'product']  # Combinations of pooling methods and modifications
    sum_methods = [(a, b) for a in pairwise_combinations for b in modifications if b == 'sum']  # Combinations of pooling methods and sum
    single_methods = [(option,) for option in pooling_options]  # Single pooling methods

    # Combine both
    experiments = product_methods + sum_methods

    data, all_seqs, emb_dict = dataset_embed('GleghornLab/optimal_temperature')
    pooler = Pooler(data, emb_dict)
    for exp in experiments:
        try:
            X_train, y_train, X_valid, y_valid, X_test, y_test = pooler.get_pooled_embeddings(exp)
            print(f"Successfully pooled with {exp}")
        except ValueError as e:
            print(e)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

"""     clf = RandomForestRegressor(n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(r2_score(y_test, y_pred)) """
