# Pooling PLM Hidden States

A research repository for systematically evaluating different pooling strategies for protein language model (PLM) hidden states across various downstream tasks. This project explores how different pooling methods and their combinations affect performance on regression and classification tasks in computational biology.

## Overview

Protein Language Models (PLMs) like ESM2 and ESM++ generate residue-level embeddings that need to be aggregated into protein-level representations for downstream tasks. This repository provides a comprehensive framework to:

- **Evaluate multiple pooling strategies**: mean, max, min, norm, median, std, var, cls, eos
- **Test pooling combinations**: pairwise combinations with concatenation, sum, or product modifications
- **Benchmark across diverse tasks**: regression and classification tasks on protein property prediction
- **Compare model architectures**: ESM2-650M, ESMplusplus_small, and other FastPLM variants

## Features

- **10 Pooling Methods**: Mean, max, min, norm, product, median, standard deviation, variance, CLS token, and EOS token pooling
- **Flexible Combinations**: Test single pooling methods, pairwise combinations, and modifications (sum/product)
- **Multiple Models**: Support for various ESM model variants via Hugging Face Transformers
- **Comprehensive Benchmarks**: 
  - Regression: optimal temperature, optimal pH, fitness prediction, enzyme kcat, stability prediction
  - Classification: MB_reg, DL2_reg, DL10_reg, cloning classification, material production
- **Result Analysis**: Automated ranking and scoring of pooling strategies across datasets

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Windows OS (as configured)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Pooling-PLM-Hidden-States
```

2. Install dependencies:
```bash
pip install torch==2.5.1 transformers==4.45.0 numpy==1.26.4 pandas scikit-learn scipy datasets matplotlib einops networkx
```

3. Install FastPLMs dependencies (if using FastPLM models):
```bash
cd FastPLMs
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from pooling_utils import Pooler
from transformers import AutoModel
from datasets import load_dataset
import torch

# Load model
model = AutoModel.from_pretrained(
    "Synthyra/ESMplusplus_small", 
    trust_remote_code=True, 
    torch_dtype=torch.float16
).cuda()

# Load dataset
data = load_dataset('GleghornLab/optimal_temperature')
all_seqs = data['train']['seqs'] + data['valid']['seqs'] + data['test']['seqs']

# Generate embeddings
emb_dict = model.embed_dataset(
    sequences=all_seqs,
    batch_size=8,
    max_len=2048,
    full_embeddings=True,
    full_precision=False,
    num_workers=0,
    sql=False,
)

# Initialize pooler
pooler = Pooler(data, emb_dict)

# Get pooled embeddings with specific pooling strategy
pooling_types = ['mean', 'max']
X_train, y_train, X_valid, y_valid, X_test, y_test = pooler.get_pooled_embeddings(
    pooling_types, 
    modification=None  # or 'sum' or 'product'
)
```

### Running Experiments

Run comprehensive experiments across multiple datasets and models:

```bash
py sum_product_pool_exp.py --n_jobs -1
```

This will:
- Test all pooling combinations across multiple datasets
- Evaluate both regression and classification tasks
- Generate CSV files with results in the `results/` directory

### Analyzing Results

Process and rank experiment results:

```bash
py score_sorter.py --input_folder results --output_file results/sorted_label_scores.csv
```

## Project Structure

```
Pooling-PLM-Hidden-States/
├── pooling_utils.py          # Core Pooler class with pooling methods
├── sum_product_pool_exp.py   # Main experiment script
├── score_sorter.py           # Result processing and ranking
├── main.py                   # Example usage script
├── pool_parti_w_esm2.py      # PoolPaRTI pooling implementation
├── FastPLMs/                 # FastPLM model implementations
│   ├── modeling_esm_plusplus.py
│   ├── modeling_fastesm.py
│   └── tests/
├── Pool_PaRTI_Benchmark/      # PoolPaRTI benchmark utilities
├── results/                   # Experiment results
│   ├── esm_probe_results/    # ESM probe experiment results
│   ├── pooling_exp_results_*/ # Pooling experiment results
│   └── sum_product/          # Sum/product pooling results
└── *.ipynb                    # Jupyter notebooks for exploration
```

## Pooling Methods

The repository supports the following pooling strategies:

| Method | Description |
|--------|-------------|
| `mean` | Average pooling across sequence length |
| `max` | Maximum value pooling |
| `min` | Minimum value pooling |
| `norm` | L2 norm pooling |
| `prod` | Product pooling (normalized by length) |
| `median` | Median value pooling |
| `std` | Standard deviation pooling |
| `var` | Variance pooling |
| `cls` | CLS token embedding (first token) |
| `eos` | EOS token embedding (last token) |

### Modifications

- **None**: Concatenate pooled embeddings (default)
- **sum**: Sum pooled embeddings element-wise
- **product**: Multiply pooled embeddings element-wise

## Supported Models

- `Synthyra/ESM2-650M`: ESM2 650M parameter model
- `Synthyra/ESMplusplus_small`: ESM++ small variant
- Other FastPLM models available via Hugging Face

See [FastPLMs/README.md](FastPLMs/README.md) for more information about supported models.

## Datasets

### Regression Tasks
- `GleghornLab/optimal_temperature`: Optimal temperature prediction
- `GleghornLab/optimal_ph`: Optimal pH prediction
- `GleghornLab/fitness_prediction`: Fitness prediction
- `GleghornLab/enzyme_kcat`: Enzyme kcat prediction
- `GleghornLab/stability_prediction`: Stability prediction

### Classification Tasks
- `GleghornLab/MB_reg`: MB regulation classification
- `GleghornLab/DL2_reg`: DL2 regulation classification
- `GleghornLab/DL10_reg`: DL10 regulation classification
- `GleghornLab/cloning_clf`: Cloning classification
- `GleghornLab/material_production`: Material production classification

## Results

Experiment results are saved in CSV format with the following metrics:
- **Regression**: Pearson correlation coefficient, R² score, and average score
- **Classification**: Matthews Correlation Coefficient (MCC)

Results are automatically sorted by performance and can be analyzed using the `score_sorter.py` script.

## Examples

### Single Pooling Method

```python
pooler = Pooler(data, emb_dict)
X_train, y_train, _, _, X_test, y_test = pooler.get_pooled_embeddings(['mean'])
```

### Pairwise Combination

```python
X_train, y_train, _, _, X_test, y_test = pooler.get_pooled_embeddings(
    ['mean', 'max'], 
    modification=None  # Concatenates to 2*dim features
)
```

### Sum Modification

```python
X_train, y_train, _, _, X_test, y_test = pooler.get_pooled_embeddings(
    ['mean', 'max'], 
    modification='sum'  # Sums to dim features
)
```

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{pooling_plm_hidden_states,
  title = {Pooling PLM Hidden States: A Comprehensive Evaluation Framework},
  author = {Your Name},
  year = {2024},
  url = {<repository-url>}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- FastPLMs for optimized protein language model implementations
- Hugging Face Transformers for model support
- GleghornLab for benchmark datasets

## Contact

For questions or issues, please open a GitHub issue or contact [your email/contact info].
