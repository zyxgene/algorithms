# Algorithms Repository - API Documentation

## Overview

This repository contains algorithm implementations for competitive programming and data science competitions, specifically:
- **Kaggle**: Liverpool Ion Switching competition solution
- **LeetCode**: Problem solving implementations

## Table of Contents

1. [Kaggle: Liverpool Ion Switching](#kaggle-liverpool-ion-switching)
   - [Configuration](#configuration)
   - [Data Processing Functions](#data-processing-functions)
   - [Feature Engineering Functions](#feature-engineering-functions)
   - [Model Functions](#model-functions)
   - [Training Functions](#training-functions)
2. [LeetCode Solutions](#leetcode-solutions)
   - [Problem 84: Largest Rectangle in Histogram](#problem-84-largest-rectangle-in-histogram)

---

## Kaggle: Liverpool Ion Switching

### Configuration

#### Global Constants
```python
EPOCHS = 190          # Number of training epochs
NNBATCHSIZE = 16     # Neural network batch size
GROUP_BATCH_SIZE = 4000  # Group batch size for data processing
SEED = 321           # Random seed for reproducibility
LR = 0.0015         # Learning rate
SPLITS = 6          # Number of cross-validation splits
```

### Data Processing Functions

#### `seed_everything(seed: int) -> None`

Sets random seeds for reproducible results across different libraries.

**Parameters:**
- `seed` (int): Random seed value

**Usage:**
```python
seed_everything(321)
```

**Description:**
Ensures reproducibility by setting seeds for:
- Python's random module
- NumPy
- TensorFlow
- Environment variables

---

#### `read_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

Loads and preprocesses the competition datasets.

**Returns:**
- `train` (pd.DataFrame): Training dataset with features and targets
- `test` (pd.DataFrame): Test dataset with features
- `sub` (pd.DataFrame): Sample submission format

**Usage:**
```python
train, test, sample_submission = read_data()
```

**Description:**
- Loads train/test data from CSV files
- Includes probability features from external RFC model
- Sets appropriate data types for memory optimization

---

#### `normalize(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`

Applies standard normalization to signal features.

**Parameters:**
- `train` (pd.DataFrame): Training dataset
- `test` (pd.DataFrame): Test dataset

**Returns:**
- `train` (pd.DataFrame): Normalized training dataset
- `test` (pd.DataFrame): Normalized test dataset

**Usage:**
```python
train_norm, test_norm = normalize(train, test)
```

**Description:**
- Calculates mean and standard deviation from training signal
- Applies z-score normalization: (x - mean) / std
- Uses training statistics for both train and test sets

---

### Feature Engineering Functions

#### `batching(df: pd.DataFrame, batch_size: int) -> pd.DataFrame`

Creates sequential groups of observations for batch processing.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `batch_size` (int): Number of observations per batch

**Returns:**
- `df` (pd.DataFrame): Dataframe with added 'group' column

**Usage:**
```python
df_batched = batching(df, batch_size=4000)
```

**Description:**
- Groups consecutive rows into batches
- Adds 'group' column with batch identifiers
- Essential for sequence-based model training

---

#### `lag_with_pct_change(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame`

Creates lag and lead features for time series analysis.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe with 'signal' and 'group' columns
- `windows` (List[int]): List of lag/lead window sizes

**Returns:**
- `df` (pd.DataFrame): Dataframe with added lag/lead features

**Usage:**
```python
df_features = lag_with_pct_change(df, windows=[1, 2, 3])
```

**Description:**
- Creates both positive and negative shifts
- Preserves group boundaries (no leakage between groups)
- Fills missing values with 0

**Generated Features:**
- `signal_shift_pos_{window}`: Lagged features
- `signal_shift_neg_{window}`: Lead features

---

#### `run_feat_engineering(df: pd.DataFrame, batch_size: int) -> pd.DataFrame`

Main feature engineering pipeline combining all transformations.

**Parameters:**
- `df` (pd.DataFrame): Raw input dataframe
- `batch_size` (int): Batch size for grouping

**Returns:**
- `df` (pd.DataFrame): Fully feature-engineered dataframe

**Usage:**
```python
df_engineered = run_feat_engineering(df, batch_size=4000)
```

**Description:**
- Applies batching
- Creates lag/lead features (windows: 1, 2, 3)
- Adds squared signal feature
- Complete preprocessing pipeline

---

#### `feature_selection(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]`

Selects features and handles missing values for model training.

**Parameters:**
- `train` (pd.DataFrame): Training dataframe
- `test` (pd.DataFrame): Test dataframe

**Returns:**
- `train` (pd.DataFrame): Cleaned training dataframe
- `test` (pd.DataFrame): Cleaned test dataframe
- `features` (List[str]): List of selected feature names

**Usage:**
```python
train_clean, test_clean, feature_list = feature_selection(train, test)
```

**Description:**
- Excludes non-feature columns: 'index', 'group', 'open_channels', 'time'
- Replaces infinite values with NaN
- Imputes missing values with cross-dataset mean

---

### Model Functions

#### `Classifier(shape_: Tuple[int, int]) -> tf.keras.Model`

Creates a WaveNet-inspired deep learning model for sequence classification.

**Parameters:**
- `shape_` (Tuple[int, int]): Input shape (sequence_length, features)

**Returns:**
- `model` (tf.keras.Model): Compiled TensorFlow model

**Usage:**
```python
model = Classifier(shape_=(4000, 8))
```

**Architecture Components:**

##### `cbr(x, out_layer, kernel, stride, dilation)`
Convolutional block with Batch Normalization and ReLU activation.

##### `wave_block(x, filters, kernel_size, n)`
WaveNet-style dilated convolution block with:
- Gated activation (tanh × sigmoid)
- Residual connections
- Progressive dilation rates: 2^0, 2^1, ..., 2^(n-1)

**Model Architecture:**
1. Initial CBR layer (64 filters, kernel=7)
2. WaveNet block (16 filters, 12 layers)
3. WaveNet block (32 filters, 8 layers)
4. WaveNet block (64 filters, 4 layers)
5. WaveNet block (128 filters, 1 layer)
6. Final CBR layer (32 filters)
7. Dropout (0.2)
8. Dense output (11 classes, softmax)

**Optimizer:** Adam with Stochastic Weight Averaging (SWA)

---

#### `lr_schedule(epoch: int) -> float`

Learning rate scheduler with step-wise decay.

**Parameters:**
- `epoch` (int): Current epoch number

**Returns:**
- `lr` (float): Learning rate for the epoch

**Usage:**
```python
callback = LearningRateScheduler(lr_schedule)
```

**Schedule:**
- Epochs 0-29: Base LR (0.0015)
- Epochs 30-39: LR/3
- Epochs 40-49: LR/5
- Epochs 50-59: LR/7
- Epochs 60-69: LR/9
- Epochs 70-79: LR/11
- Epochs 80-89: LR/13
- Epochs 90+: LR/100

---

### Training Functions

#### `MacroF1(Callback)`

Custom Keras callback for monitoring macro F1-score during training.

**Constructor Parameters:**
- `model`: Keras model instance
- `inputs`: Validation input data
- `targets`: Validation target data (one-hot encoded)

**Usage:**
```python
f1_callback = MacroF1(model, valid_x, valid_y)
model.fit(..., callbacks=[f1_callback])
```

**Methods:**
- `on_epoch_end(epoch, logs)`: Calculates and prints macro F1-score

---

#### `run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size) -> None`

Main cross-validation training function using GroupKFold.

**Parameters:**
- `train` (pd.DataFrame): Training dataset
- `test` (pd.DataFrame): Test dataset
- `splits` (int): Number of CV splits
- `batch_col` (str): Column name for grouping ('group')
- `feats` (List[str]): Feature column names
- `sample_submission` (pd.DataFrame): Submission format
- `nn_epochs` (int): Number of training epochs
- `nn_batch_size` (int): Training batch size

**Usage:**
```python
run_cv_model_by_batch(
    train=train_data,
    test=test_data,
    splits=5,
    batch_col='group',
    feats=feature_names,
    sample_submission=submission_df,
    nn_epochs=190,
    nn_batch_size=16
)
```

**Process:**
1. Sets up GroupKFold cross-validation (5 folds)
2. Converts data to 3D arrays (groups × sequence × features)
3. One-hot encodes target classes (11 classes)
4. Trains model on each fold
5. Generates out-of-fold predictions
6. Averages test predictions across folds
7. Saves submission file

**Output:**
- Prints training progress and F1-scores
- Saves 'submission_files.csv'

---

#### `run_everything() -> None`

Main orchestration function that executes the complete pipeline.

**Usage:**
```python
run_everything()
```

**Pipeline Steps:**
1. **Data Loading**: Read train/test/submission files
2. **Normalization**: Apply standard scaling
3. **Feature Engineering**: Create lag/lead features and batches
4. **Feature Selection**: Select relevant features and handle missing values
5. **Model Training**: Run cross-validation with WaveNet model
6. **Submission**: Generate final predictions

**Output:**
- Console logs of progress
- 'submission_files.csv' with predictions

---

## LeetCode Solutions

### Problem 84: Largest Rectangle in Histogram

#### `HistogramArea(arr: List[int]) -> int`

Finds the largest rectangular area in a histogram using stack-based algorithm.

**Parameters:**
- `arr` (List[int]): Array of non-negative integers representing bar heights

**Returns:**
- `max_area` (int): Maximum rectangular area possible

**Usage:**
```python
heights = [2, 1, 5, 6, 2, 3]
max_area = HistogramArea(heights)
print(max_area)  # Output: 10
```

**Algorithm:**
1. **Stack Approach**: Uses monotonic stack to track indices
2. **Height Processing**: For each bar, finds maximum rectangle with that bar as the shortest
3. **Width Calculation**: Uses stack to determine left and right boundaries
4. **Time Complexity**: O(n) - each element pushed/popped once
5. **Space Complexity**: O(n) - stack storage

**Step-by-Step Process:**
1. Append 0 to array (sentinel value)
2. Initialize stack with [-1] (boundary marker)
3. For each position:
   - While current height < stack top height:
     - Pop height and calculate rectangle area
     - Width = current_index - new_stack_top - 1
   - Push current index
4. Return maximum area found

**Example Walkthrough:**
```python
arr = [2, 1, 5, 6, 2, 3]
# Step by step:
# i=0: stack=[-1,0], height=2
# i=1: height=1 < 2, pop 0, area=2*1=2, stack=[-1,1]
# i=2: stack=[-1,1,2], height=5
# i=3: stack=[-1,1,2,3], height=6
# i=4: height=2 < 6, pop 3, area=6*1=6
#      height=2 < 5, pop 2, area=5*2=10
# i=5: stack=[-1,1,4,5], height=3
# i=6: height=0, clean stack, final max_area=10
```

**Edge Cases:**
- Empty array: Returns 0
- Single element: Returns element value
- Increasing sequence: Max area at the end
- Decreasing sequence: Max area at the beginning
- All same height: Area = height × length

---

## Installation & Dependencies

### Kaggle Solution Requirements
```bash
pip install tensorflow==2.x
pip install tensorflow_addons==0.9.1
pip install pandas numpy scikit-learn
```

### LeetCode Solution Requirements
- Python 3.x (standard library only)

---

## Usage Examples

### Complete Kaggle Pipeline
```python
# Run the complete competition solution
run_everything()
```

### Individual Components
```python
# Data preprocessing
train, test, sub = read_data()
train, test = normalize(train, test)

# Feature engineering
train = run_feat_engineering(train, GROUP_BATCH_SIZE)
test = run_feat_engineering(test, GROUP_BATCH_SIZE)

# Model training
train, test, features = feature_selection(train, test)
run_cv_model_by_batch(train, test, SPLITS, 'group', features, sub, EPOCHS, NNBATCHSIZE)
```

### LeetCode Problem Solving
```python
# Test with sample input
test_heights = [2, 1, 5, 6, 2, 3]
result = HistogramArea(test_heights)
print(f"Maximum rectangle area: {result}")
```

---

## Performance Notes

### Kaggle Solution
- **Memory Usage**: ~8GB RAM recommended for GROUP_BATCH_SIZE=4000
- **Training Time**: ~2-3 hours on GPU (190 epochs)
- **Model Size**: ~50MB for the WaveNet architecture
- **Validation**: Uses GroupKFold to prevent data leakage

### LeetCode Solution
- **Time Complexity**: O(n) linear time
- **Space Complexity**: O(n) for the stack
- **Optimal**: Uses monotonic stack for efficient processing

---

## Contributing

When adding new algorithms:
1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include usage examples
4. Update this documentation
5. Add appropriate test cases