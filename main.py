from itertools import combinations
from pooling_utils import Pooler
import torch
from transformers import AutoModel
from datasets import load_dataset

data = load_dataset('GleghornLab/optimal_temperature')
all_seqs = data['train']['seqs'] + data['valid']['seqs'] + data['test']['seqs']
len(all_seqs)

model = AutoModel.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True, torch_dtype=torch.float16).cuda()

pooling_options = ['mean', 'max', 'min', 'norm', 'prod', 'median', 'std', 'var', 'cls', 'eos']

experiments = list(combinations(pooling_options, 2)) # for the first 45, number of features 960 * 2
for pooling_option in pooling_options:
    experiments.append((pooling_option,)) # for the last 10, number of features 960

print(experiments)
print(len(experiments))

emb_dict = model.embed_dataset(
    sequences=all_seqs, # list of protein strings
    batch_size=8, # embedding batch size
    max_len=2048, # truncate to max_len
    full_embeddings=True, # return residue-wise embeddings
    full_precision=False, # store as float32
    pooling_type='mean', # use mean pooling if protein-wise embeddings
    num_workers=0, # data loading num workers
    sql=False, # return dictionary of sequences and embeddings
)

for exp in experiments:
    Pooler.__init__(data, emb_dict)
    Pooler.get_pooled_embeddings(pooling_options)