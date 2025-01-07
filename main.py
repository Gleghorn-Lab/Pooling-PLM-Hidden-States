from itertools import combinations
from pooling_utils import Pooler


pooling_options = ['mean', 'max', 'min', 'norm', 'prod', 'median', 'std', 'var', 'cls', 'eos']

experiments = list(combinations(pooling_options, 2)) # for the first 45, number of features 960 * 2
for pooling_option in pooling_options:
    experiments.append((pooling_option,)) # for the last 10, number of features 960

print(experiments)
print(len(experiments))