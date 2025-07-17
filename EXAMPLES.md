# Examples and Usage Guide

## Table of Contents
1. [Kaggle Examples](#kaggle-examples)
2. [LeetCode Examples](#leetcode-examples)
3. [Advanced Usage Patterns](#advanced-usage-patterns)
4. [Testing and Validation](#testing-and-validation)

---

## Kaggle Examples

### Example 1: Complete Pipeline Execution

```python
# Simple one-line execution
run_everything()

# This will:
# 1. Load data from /kaggle/input/
# 2. Normalize signals 
# 3. Create features (lags, leads, signal^2)
# 4. Train WaveNet model with 5-fold CV
# 5. Generate submission_files.csv
```

### Example 2: Custom Configuration

```python
# Modify global parameters before running
EPOCHS = 100          # Reduce for faster training
NNBATCHSIZE = 32     # Increase if you have more memory
GROUP_BATCH_SIZE = 2000  # Smaller groups for less memory usage
LR = 0.001           # Lower learning rate

# Then run the pipeline
run_everything()
```

### Example 3: Step-by-Step with Custom Processing

```python
import pandas as pd
import numpy as np

# Step 1: Load and examine data
train, test, submission = read_data()
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['open_channels'].value_counts().sort_index()}")

# Step 2: Apply normalization
train_norm, test_norm = normalize(train, test)
print(f"Signal mean after normalization: {train_norm['signal'].mean():.6f}")
print(f"Signal std after normalization: {train_norm['signal'].std():.6f}")

# Step 3: Feature engineering with custom batch size
train_feat = run_feat_engineering(train_norm, batch_size=4000)
test_feat = run_feat_engineering(test_norm, batch_size=4000)

# Examine created features
feature_cols = [col for col in train_feat.columns if 'signal' in col]
print(f"Created features: {feature_cols}")

# Step 4: Feature selection and cleaning
train_clean, test_clean, features = feature_selection(train_feat, test_feat)
print(f"Selected {len(features)} features for training")

# Step 5: Train model with custom parameters
run_cv_model_by_batch(
    train=train_clean,
    test=test_clean,
    splits=5,
    batch_col='group',
    feats=features,
    sample_submission=submission,
    nn_epochs=150,      # Custom epoch count
    nn_batch_size=24    # Custom batch size
)
```

### Example 4: Model Architecture Exploration

```python
# Create and examine model architecture
model = Classifier(shape_=(4000, 8))
model.summary()

# Print model configuration
print(f"Total parameters: {model.count_params():,}")
print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

# Visualize learning rate schedule
epochs = range(200)
learning_rates = [lr_schedule(epoch) for epoch in epochs]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(epochs, learning_rates)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.grid(True)
plt.show()
```

### Example 5: Custom Feature Engineering

```python
def custom_feature_engineering(df, batch_size):
    """Enhanced feature engineering with additional features"""
    # Standard features
    df = run_feat_engineering(df, batch_size)
    
    # Add custom features
    df['signal_abs'] = df['signal'].abs()
    df['signal_sqrt'] = np.sqrt(df['signal'].abs())
    df['signal_log'] = np.log1p(df['signal'].abs())
    
    # Rolling statistics within groups
    for window in [10, 50, 100]:
        df[f'signal_rolling_mean_{window}'] = df.groupby('group')['signal'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        df[f'signal_rolling_std_{window}'] = df.groupby('group')['signal'].rolling(window, min_periods=1).std().reset_index(0, drop=True)
    
    return df

# Use custom feature engineering
train_custom = custom_feature_engineering(train_norm, GROUP_BATCH_SIZE)
test_custom = custom_feature_engineering(test_norm, GROUP_BATCH_SIZE)
```

---

## LeetCode Examples

### Example 1: Basic Usage

```python
# Test case 1: Standard histogram
heights = [2, 1, 5, 6, 2, 3]
result = HistogramArea(heights)
print(f"Input: {heights}")
print(f"Maximum rectangle area: {result}")  # Output: 10

# Test case 2: Single bar
heights = [5]
result = HistogramArea(heights)
print(f"Input: {heights}")
print(f"Maximum rectangle area: {result}")  # Output: 5

# Test case 3: Increasing heights
heights = [1, 2, 3, 4, 5]
result = HistogramArea(heights)
print(f"Input: {heights}")
print(f"Maximum rectangle area: {result}")  # Output: 9
```

### Example 2: Edge Cases Testing

```python
# Edge case 1: Empty array
heights = []
result = HistogramArea(heights)
print(f"Empty array result: {result}")  # Output: 0

# Edge case 2: All same height
heights = [3, 3, 3, 3]
result = HistogramArea(heights)
print(f"Same height result: {result}")  # Output: 12

# Edge case 3: Decreasing heights
heights = [5, 4, 3, 2, 1]
result = HistogramArea(heights)
print(f"Decreasing heights result: {result}")  # Output: 9

# Edge case 4: Single tall bar
heights = [0, 0, 0, 10, 0, 0, 0]
result = HistogramArea(heights)
print(f"Single tall bar result: {result}")  # Output: 10
```

### Example 3: Algorithm Visualization

```python
def HistogramAreaWithVisualization(arr):
    """Enhanced version with step-by-step visualization"""
    print(f"Processing histogram: {arr}")
    
    n = len(arr)
    arr.append(0)  # Sentinel
    stack = [-1]
    max_area = 0
    
    for i in range(n + 1):
        print(f"Step {i}: height={arr[i]}, stack={stack}")
        
        while arr[i] < arr[stack[-1]]:
            h = arr[stack.pop()]
            w = i - stack[-1] - 1
            area = h * w
            max_area = max(max_area, area)
            print(f"  Popped height {h}, width {w}, area {area}, max_area {max_area}")
        
        stack.append(i)
        print(f"  Stack after push: {stack}")
    
    return max_area

# Visualize the algorithm
heights = [2, 1, 5, 6, 2, 3]
result = HistogramAreaWithVisualization(heights)
```

### Example 4: Performance Testing

```python
import time
import random

def performance_test():
    """Test performance on different input sizes"""
    sizes = [100, 1000, 10000, 100000]
    
    for size in sizes:
        # Generate random histogram
        heights = [random.randint(0, 1000) for _ in range(size)]
        
        start_time = time.time()
        result = HistogramArea(heights)
        end_time = time.time()
        
        print(f"Size: {size:6d}, Result: {result:8d}, Time: {end_time - start_time:.4f}s")

performance_test()
```

---

## Advanced Usage Patterns

### Pattern 1: Kaggle Ensemble Method

```python
def ensemble_predictions():
    """Create ensemble of models with different configurations"""
    predictions = []
    
    configs = [
        {'epochs': 150, 'batch_size': 16, 'lr': 0.0015},
        {'epochs': 200, 'batch_size': 24, 'lr': 0.001},
        {'epochs': 180, 'batch_size': 32, 'lr': 0.002}
    ]
    
    for i, config in enumerate(configs):
        print(f"Training model {i+1} with config: {config}")
        
        # Modify global variables
        global EPOCHS, NNBATCHSIZE, LR
        EPOCHS = config['epochs']
        NNBATCHSIZE = config['batch_size']
        LR = config['lr']
        
        # Train model
        run_everything()
        
        # Load predictions (assuming they're saved)
        pred = pd.read_csv('submission_files.csv')
        predictions.append(pred['open_channels'].values)
    
    # Ensemble by majority voting
    ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
    
    # Save ensemble result
    submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
    submission['open_channels'] = ensemble_pred
    submission.to_csv('ensemble_submission.csv', index=False)
```

### Pattern 2: Hyperparameter Optimization

```python
def hyperparameter_search():
    """Grid search for optimal hyperparameters"""
    from itertools import product
    
    # Define parameter grid
    param_grid = {
        'learning_rates': [0.001, 0.0015, 0.002],
        'batch_sizes': [16, 24, 32],
        'group_sizes': [2000, 4000, 6000]
    }
    
    best_score = -1
    best_params = None
    
    for lr, bs, gs in product(param_grid['learning_rates'], 
                             param_grid['batch_sizes'], 
                             param_grid['group_sizes']):
        
        print(f"Testing: LR={lr}, Batch={bs}, Group={gs}")
        
        # Set parameters
        global LR, NNBATCHSIZE, GROUP_BATCH_SIZE
        LR = lr
        NNBATCHSIZE = bs
        GROUP_BATCH_SIZE = gs
        
        # Quick validation (reduced epochs)
        global EPOCHS
        EPOCHS = 50
        
        # Run pipeline and capture score
        # Note: You'd need to modify run_everything() to return validation score
        score = run_everything()  # Assume this returns validation F1 score
        
        if score > best_score:
            best_score = score
            best_params = {'lr': lr, 'batch_size': bs, 'group_size': gs}
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
```

### Pattern 3: Custom Validation Strategy

```python
def time_series_validation():
    """Implement time-aware validation for better evaluation"""
    
    # Load data
    train, test, submission = read_data()
    train, test = normalize(train, test)
    train = run_feat_engineering(train, GROUP_BATCH_SIZE)
    
    # Time-based split (assuming 'time' column exists)
    train_sorted = train.sort_values('time')
    split_point = int(len(train_sorted) * 0.8)
    
    train_time = train_sorted.iloc[:split_point]
    val_time = train_sorted.iloc[split_point:]
    
    print(f"Training samples: {len(train_time)}")
    print(f"Validation samples: {len(val_time)}")
    
    # Train on time-split data
    # ... (implement custom training loop)
```

---

## Testing and Validation

### Unit Tests Example

```python
import unittest

class TestAlgorithms(unittest.TestCase):
    
    def test_histogram_area_basic(self):
        """Test basic histogram area calculation"""
        self.assertEqual(HistogramArea([2, 1, 5, 6, 2, 3]), 10)
        self.assertEqual(HistogramArea([1, 1, 1, 1]), 4)
        self.assertEqual(HistogramArea([5]), 5)
    
    def test_histogram_area_edge_cases(self):
        """Test edge cases"""
        self.assertEqual(HistogramArea([]), 0)
        self.assertEqual(HistogramArea([0, 0, 0]), 0)
        self.assertEqual(HistogramArea([1, 0, 1]), 1)
    
    def test_feature_engineering(self):
        """Test feature engineering functions"""
        # Create sample data
        df = pd.DataFrame({
            'signal': [1, 2, 3, 4, 5],
            'time': [0, 1, 2, 3, 4]
        })
        
        # Test batching
        df_batched = batching(df, batch_size=2)
        self.assertIn('group', df_batched.columns)
        
        # Test lag features
        df_lag = lag_with_pct_change(df_batched, [1])
        self.assertIn('signal_shift_pos_1', df_lag.columns)
        self.assertIn('signal_shift_neg_1', df_lag.columns)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
def test_kaggle_pipeline():
    """Test the complete Kaggle pipeline with small dataset"""
    
    # Create mock data
    mock_train = pd.DataFrame({
        'time': np.arange(1000),
        'signal': np.random.randn(1000),
        'open_channels': np.random.randint(0, 11, 1000)
    })
    
    mock_test = pd.DataFrame({
        'time': np.arange(500),
        'signal': np.random.randn(500)
    })
    
    # Test individual components
    try:
        # Test normalization
        train_norm, test_norm = normalize(mock_train.copy(), mock_test.copy())
        assert abs(train_norm['signal'].mean()) < 0.01
        assert abs(train_norm['signal'].std() - 1.0) < 0.01
        
        # Test feature engineering
        train_feat = run_feat_engineering(train_norm, batch_size=100)
        assert 'group' in train_feat.columns
        assert 'signal_2' in train_feat.columns
        
        print("All pipeline components working correctly!")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")

test_kaggle_pipeline()
```

### Performance Benchmarks

```python
def benchmark_algorithms():
    """Benchmark algorithm performance"""
    import time
    
    # Benchmark histogram algorithm
    test_cases = [
        ([1] * 1000, "Flat histogram"),
        (list(range(1000)), "Increasing histogram"),
        (list(range(1000, 0, -1)), "Decreasing histogram"),
        ([random.randint(1, 1000) for _ in range(1000)], "Random histogram")
    ]
    
    for heights, description in test_cases:
        start_time = time.time()
        result = HistogramArea(heights)
        end_time = time.time()
        
        print(f"{description:20s}: {result:8d} in {end_time - start_time:.4f}s")

benchmark_algorithms()
```

This comprehensive documentation provides:

1. **Complete API documentation** with detailed function descriptions, parameters, return values, and usage examples
2. **Quick reference guide** for fast lookup of function signatures
3. **Practical examples** covering basic usage, advanced patterns, and edge cases
4. **Testing strategies** including unit tests and integration tests
5. **Performance considerations** and optimization tips

The documentation is organized for different user needs:
- Beginners can start with the basic examples
- Advanced users can explore the customization patterns
- Developers can reference the API documentation for implementation details
- The quick reference provides fast lookup during development