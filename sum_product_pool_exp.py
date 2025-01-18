import torch
import numpy as np
import pandas as pd
from transformers import AutoModel
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from itertools import combinations
from typing import Tuple
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, make_scorer
from sklearn.metrics import matthews_corrcoef


pooling_options = ['mean', 'max', 'min', 'norm', 'median', 'std', 'var', 'cls', 'eos']
modifications = ['None', 'sum', 'product']

datasets_regression = ['GleghornLab/optimal_temperature',
                        'GleghornLab/optimal_ph',
                        'GleghornLab/TS_reg',
                        'GleghornLab/fitness_prediction',
                        'GleghornLab/enzyme_kcat'
                        ]

datasets_classification = ['GleghornLab/MB_reg',
                            'GleghornLab/DL2_reg',
                            'GleghornLab/DL10_reg',
                            'GleghornLab/cloning_clf'
                            ]

models = ["Synthyra/ESM2-650M", "Synthyra/ESMplusplus_small"]


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
            'median': self.median_pooling,
            'std': self.std_pooling,
            'var': self.var_pooling,
            'cls': self.cls_pooling,
            'eos': self.eos_pooling,
        }

    def mean_pooling(self, emb): # (L, d) -> (d,)
        return emb.mean(dim=0)

    def max_pooling(self, emb): # (L, d) -> (d,)
        return emb.max(dim=0).values

    def min_pooling(self, emb): # (L, d) -> (d,)
        return emb.min(dim=0).values

    def norm_pooling(self, emb): # (L, d) -> (d,)
        return emb.norm(dim=0, p=2)

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


def pearson_metric(y_true, y_pred):
    return pearsonr(y_true, y_pred).correlation
pearson_metric.__name__ = "pearson_metric"
pearson_scorer = make_scorer(pearson_metric, greater_is_better=True)


def r2_metric(y_true, y_pred):
    return r2_score(y_true, y_pred)
r2_metric.__name__ = "r2_metric"
r2_scorer = make_scorer(r2_metric, greater_is_better=True)


def mcc_metric(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)
mcc_metric.__name__ = "mcc_metric"
mcc_scorer = make_scorer(mcc_metric, greater_is_better=True)


def dataset_embed(model, dataset_name):
    data = load_dataset(dataset_name)
    all_seqs = list(set(data['train']['seqs'] + data['valid']['seqs'] + data['test']['seqs']))
    print(f"Number of sequences: {len(all_seqs)}")
    emb_dict = model.embed_dataset(
        sequences=all_seqs, # list of protein strings
        batch_size=8, # embedding batch size
        max_len=8192, # truncate to max_len
        full_embeddings=True, # return residue-wise embeddings
        full_precision=False, # store as float32
        num_workers=0, # data loading num workers
        sql=False, # return dictionary of sequences and embeddings
        )
    return data, all_seqs, emb_dict


# Generate experiments
def create_experiments(pooling_options, modifications):
    pairwise_combinations = list(combinations(pooling_options, 2))  # Add 'None' to pairwise combinations
    comb_methods = [(a, b) for a in pairwise_combinations for b in modifications if b == 'None']  # 'None' methods
    product_methods = [(a, b) for a in pairwise_combinations for b in modifications if b == 'product']  # 'product' methods
    sum_methods = [(a, b) for a in pairwise_combinations for b in modifications if b == 'sum']  # 'sum' methods
    single_methods = [((option,), 'None') for option in pooling_options]

    # Combine all into experiments
    experiments = comb_methods + product_methods + sum_methods + single_methods
    return experiments    


def regression_results(datasets_regression, model, model_name):
    for dataset_name in datasets_regression:
        data, all_seqs, emb_dict = dataset_embed(model, dataset_name)
        experiments = create_experiments(pooling_options, modifications)
        pooler = Pooler(data, emb_dict)
        results = {}
        print(f"Dataset: {dataset_name}")
        for exp, modification in experiments:
            experiment = exp, modification
            try:
                X_train, y_train, X_valid, y_valid, X_test, y_test = pooler.get_pooled_embeddings(exp, modification)
                print(f"Successfully pooled with {exp}")
            except ValueError as e:
                print(e)
            test_model = RandomForestRegressor(
                criterion='poisson',
                max_depth=100,
                max_features='sqrt',
                min_samples_leaf=4,
                min_samples_split=5,
                random_state = 42,
                oob_score=False,
                warm_start=False,
                ccp_alpha=0.0,
                bootstrap=False,
                n_jobs=-1
            )
            test_model.fit(X_train, y_train)
            score_pearson = pearson_scorer(test_model, X_test, y_test)
            score_r2 = r2_scorer(test_model, X_test, y_test)
            score_avg = (score_pearson + score_r2) / 2
            print(f"Experiment: {exp}")
            print(f"Modification: {modification}")
            print(f"Pearson Score Tested: {score_pearson}")
            print(f"R2 Score Tested: {score_r2}")
            print(f"Average Score: {score_avg}")
            results[experiment] = {'pearson': score_pearson, 'r2': score_r2, 'score_average': score_avg}
        df = pd.DataFrame.from_dict(results)
        df.sort_values(by='score_average', axis=1, ascending=False, inplace=True)
        df.to_csv(f'{model_name[9:]}_results_{dataset_name[12:]}_regr.csv')


def classification_results(datasets_classification, model, model_name):
    for dataset_name in datasets_classification:
        data, all_seqs, emb_dict = dataset_embed(model, dataset_name)
        experiments = create_experiments(pooling_options, modifications)
        pooler = Pooler(data, emb_dict)
        results = {}
        print(f"Dataset: {dataset_name}")
        for exp, modification in experiments:
            experiment = exp, modification
            try:
                X_train, y_train, X_valid, y_valid, X_test, y_test = pooler.get_pooled_embeddings(exp, modification)
                print(f"Successfully pooled with {exp}")
            except ValueError as e:
                print(e)
            test_model = RandomForestClassifier(
                bootstrap=True,
                ccp_alpha=0.0,
                class_weight=None,
                criterion='log_loss',
                max_depth=30,
                max_features='sqrt',
                min_samples_leaf=10,
                min_samples_split=10,
                n_estimators=100,
                oob_score=False,
                warm_start=False,
                random_state = 42,
                n_jobs=-1
            )
            test_model.fit(X_train, y_train)
            score_mcc = mcc_scorer(test_model, X_test, y_test)
            print(f"Experiment: {exp}")
            print(f"Modification: {modification}")
            print(f"MCC Score Tested: {score_mcc}")
            results[experiment] = {'mcc': score_mcc}
        df = pd.DataFrame.from_dict(results)
        df.sort_values(by='mcc', axis=1, ascending=False, inplace=True)
        df.to_csv(f'{model_name[9:]}_results_{dataset_name[12:]}_class.csv')


def test_multiple_models(models):
    for model_name in models:
        print(f'Testing all possible combinations for all datasets in: {model_name}')
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda()
        regression_results(datasets_regression, model, model_name)
        classification_results(datasets_classification, model, model_name)
        print(f'Completed and downloaded results for: {model_name}')
        model.cpu()
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # Define pooling options, modifications, datasets, and models to use
    test_multiple_models(models)
