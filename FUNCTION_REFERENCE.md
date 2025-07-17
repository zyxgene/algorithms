# Function Reference - Quick Lookup

## Kaggle: Liverpool Ion Switching (`kaggle/liverpool-ion-switching.py`)

### Data Processing
| Function | Signature | Description |
|----------|-----------|-------------|
| `seed_everything` | `(seed: int) -> None` | Sets random seeds for reproducibility |
| `read_data` | `() -> Tuple[DataFrame, DataFrame, DataFrame]` | Loads train/test/submission data |
| `normalize` | `(train: DataFrame, test: DataFrame) -> Tuple[DataFrame, DataFrame]` | Applies z-score normalization |

### Feature Engineering
| Function | Signature | Description |
|----------|-----------|-------------|
| `batching` | `(df: DataFrame, batch_size: int) -> DataFrame` | Creates sequential groups |
| `lag_with_pct_change` | `(df: DataFrame, windows: List[int]) -> DataFrame` | Creates lag/lead features |
| `run_feat_engineering` | `(df: DataFrame, batch_size: int) -> DataFrame` | Complete feature engineering pipeline |
| `feature_selection` | `(train: DataFrame, test: DataFrame) -> Tuple[DataFrame, DataFrame, List[str]]` | Feature selection and cleaning |

### Model Architecture
| Function | Signature | Description |
|----------|-----------|-------------|
| `Classifier` | `(shape_: Tuple[int, int]) -> tf.keras.Model` | WaveNet-inspired deep learning model |
| `lr_schedule` | `(epoch: int) -> float` | Learning rate scheduler |

### Training & Evaluation
| Function | Signature | Description |
|----------|-----------|-------------|
| `MacroF1.__init__` | `(model, inputs, targets)` | F1-score monitoring callback |
| `run_cv_model_by_batch` | `(train, test, splits, batch_col, feats, submission, epochs, batch_size) -> None` | Cross-validation training |
| `run_everything` | `() -> None` | Complete pipeline orchestration |

## LeetCode Solutions (`leetcode/leetcode84.py`)

### Problem 84: Largest Rectangle in Histogram
| Function | Signature | Description |
|----------|-----------|-------------|
| `HistogramArea` | `(arr: List[int]) -> int` | Stack-based largest rectangle algorithm |

## Constants & Configuration

### Kaggle Global Constants
```python
EPOCHS = 190          # Training epochs
NNBATCHSIZE = 16     # Neural network batch size  
GROUP_BATCH_SIZE = 4000  # Data grouping batch size
SEED = 321           # Random seed
LR = 0.0015         # Base learning rate
SPLITS = 6          # Cross-validation splits
```

## Quick Usage Patterns

### Kaggle - End-to-End Pipeline
```python
run_everything()  # Complete solution
```

### Kaggle - Step-by-Step
```python
# 1. Data loading and normalization
train, test, sub = read_data()
train, test = normalize(train, test)

# 2. Feature engineering
train = run_feat_engineering(train, GROUP_BATCH_SIZE)
test = run_feat_engineering(test, GROUP_BATCH_SIZE)

# 3. Model training
train, test, features = feature_selection(train, test)
run_cv_model_by_batch(train, test, SPLITS, 'group', features, sub, EPOCHS, NNBATCHSIZE)
```

### LeetCode - Problem Solving
```python
heights = [2, 1, 5, 6, 2, 3]
max_area = HistogramArea(heights)  # Returns: 10
```