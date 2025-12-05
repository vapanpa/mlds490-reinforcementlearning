# MLDS 490 - Assignment 4: AutoML Hyperparameter Tuning

Neural network hyperparameter optimization using evolutionary and probabilistic search methods on the Federated EMNIST digit classification task.

## Overview

This project compares two automated hyperparameter tuning approaches:

| Method | Description |
|--------|-------------|
| **Genetic Algorithm** | Population-based evolutionary search with roulette selection, one-point crossover, and age-based replacement |
| **Bayesian Optimization** | Gaussian Process surrogate model with acquisition function-guided search |

### Hyperparameters Optimized
- **Mini-batch size**: Integer in range [16, 1024]
- **Hidden layer activation**: One of {ReLU, Sigmoid, Tanh}

## Project Structure

```
HW4/
├── HW4.ipynb          # Main notebook with all implementations
├── data/
│   ├── train_data.npy # Federated EMNIST training data
│   └── test_data.npy  # Federated EMNIST test data
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Setup Instructions

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Data Files

Ensure `train_data.npy` and `test_data.npy` are located in the `data/` subdirectory. The dataset should be in the federated format with client dictionaries containing `images` and `labels` keys.

## Running the Notebook

Open `HW4.ipynb` in Jupyter Notebook or JupyterLab and execute cells sequentially:

```bash
jupyter notebook HW4.ipynb
```

**Note:** GPU acceleration is automatically enabled if CUDA is available.

## Implementation Details

### Neural Network Architecture
- Input layer: 784 features (28×28 flattened images)
- Hidden layer: 256 units with configurable activation
- Output layer: 10 classes (digits 0-9)
- Optimizer: Stochastic Gradient Descent (lr=0.01)
- Loss: Cross-Entropy

### Genetic Algorithm Configuration
| Parameter | Value |
|-----------|-------|
| Population size | 20 |
| Generations | 20 |
| Maximum lifespan | 5 generations |
| Mutation rate | 15% |
| Selection | Roulette wheel |
| Crossover | One-point |

### Bayesian Optimization Configuration
| Parameter | Value |
|-----------|-------|
| Initial random points | 10 |
| Optimization iterations | 40 |
| Surrogate model | Gaussian Process |

## Evaluation Metric

All models are evaluated using **macro-averaged F1 score**, which weights each class equally regardless of sample frequency.

## Expected Outputs

1. **GA Evolution Plot**: Average and best fitness scores per generation
2. **Training Curves**: F1 score vs. epochs for both optimization methods
3. **Final Test Results**: Test F1 scores for models trained with optimal hyperparameters

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` for data files | Verify data files are in `data/` subdirectory |
| CUDA out of memory | Reduce batch size or use CPU (`COMPUTE_DEVICE = torch.device("cpu")`) |
| Slow execution | Reduce `generation_limit` or `n_iterations` for faster (less accurate) results |

## License

Academic use only - Northwestern University MLDS 490 coursework.

