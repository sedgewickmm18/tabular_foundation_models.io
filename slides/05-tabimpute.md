<!-- Main Slide: TabImpute Overview -->
## TabImpute
### Intelligent Missing Value Imputation

**A Python library for intelligent imputation of missing values in tabular data**

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem;">

<div>

### 🎯 Key Features
- Numeric & categorical imputation
- Confidence scoring
- Chunk-based processing
- Device-aware (CPU/CUDA/MPS)

</div>

<div>

### 🚀 Quick Start
```python
from tabimpute.tabimpute_v2 import TabImputeV2

imputer = TabImputeV2(device='cpu')
X_imputed = imputer.impute(X_array)
```

</div>

</div>

Note:
TabImpute provides a simple interface to fill missing values using pre-trained
tabular foundation models from Hugging Face. It handles both numeric and categorical
columns with built-in uncertainty quantification.

--

<!-- Vertical Slide: Installation & Prerequisites -->
## Installation & Setup

### Installation

```bash
pip install tabimpute
```

Or from source:
```bash
git clone https://github.com/<your-org>/tabimpute.git
cd tabimpute
pip install -e .
```

### Prerequisites

**Hugging Face Token Required**

```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token_here"
```

Generate token at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Note:
TabImpute uses gated tabular models hosted on Hugging Face, requiring authentication
before running any imputation tasks.

--

<!-- Vertical Slide: Core API -->
## API Reference

### TabImputeV2 (Numeric Data)

```python
from tabimpute.tabimpute_v2 import TabImputeV2

imputer = TabImputeV2(device='cpu', verbose=False)
X_imputed = imputer.impute(X_array)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device`  | str  | `'cpu'` | Device: `'cpu'`, `'cuda'`, or `'mps'` |
| `verbose` | bool | `False` | Print progress messages |

### ImputePFN (Alternative)

```python
from tabimpute.interface import ImputePFN

imputer = ImputePFN(device='cpu', verbose=True)
X_imputed = imputer.impute(X_array)

# Split large arrays into chunks
chunks, starts, ends = imputer.split_into_chunks(X_array, chunk_size=1000)
```

Note:
TabImputeV2 is the main imputer for numeric data, while ImputePFN provides
compatibility with chunk-based workflows for larger datasets.

--

<!-- Vertical Slide: Categorical Imputation -->
## Categorical Imputation

### TabImputeCategorical

```python
from tabimpute.interface import TabImputeCategorical

categorical = TabImputeCategorical(device='cpu')
X_cat_imputed = categorical.impute(
    X_cat_array, 
    categorical_columns=[0, 1]
)
```

| Parameter             | Type       | Description |
|-----------------------|------------|-------------|
| `categorical_columns` | list[int]  | Column indices to treat as categorical |

### Use Cases
- String/object columns
- Categorical features
- Mixed data types

Note:
TabImputeCategorical handles non-numeric columns separately, using specialized
models trained for categorical data imputation.

--

<!-- Vertical Slide: Small Dataset Example -->
## Small Datasets with Confidence Scores

### Basic Imputation

```python
import numpy as np
import pandas as pd
from tabimpute.tabimpute_v2 import TabImputeV2

# Load data with missing values
df = pd.read_csv('heart.csv')
X, y = df.drop(columns=['target']), df["target"]

# Introduce missing values
X_test_missing = X.copy()
n_samples, n_features = X_test_missing.shape
missing_fraction = 0.25
n_missing = int(n_samples * missing_fraction)

for col_idx in range(3):
    missing_indices = np.random.choice(n_samples, n_missing, replace=False)
    X_test_missing.iloc[missing_indices, col_idx] = np.nan

# Impute
imputer = TabImputeV2(device='cpu', verbose=False)
X_array = np.array(X_test_missing, dtype=np.float32)
X_imputed = imputer.impute(X_array)
```

Note:
For smaller datasets, you can run imputation directly without chunking.
The model handles missing values efficiently for datasets under 10,000 rows.

--

<!-- Vertical Slide: Confidence Scoring -->
## Confidence & Uncertainty Quantification

### Extract Confidence Scores

```python
import torch

def impute_with_uncertainty(imputer, X, verbose=True):
    """
    Impute missing values and return per-value confidence scores
    derived from the model's predicted distribution variance.
    
    Returns: imputed_values, confidence_scores, uncertainty_std, variance
    """
    means = np.nanmean(X, axis=0)
    stds  = np.nanstd(X, axis=0)
    stds  = np.where(stds == 0, 1, stds)
    
    X_norm = (X - means) / (stds + 1e-16)
    
    # Run model
    X_tensor = torch.from_numpy(X_norm).unsqueeze(0).to(torch.bfloat16).to(imputer.device)
    with torch.no_grad():
        preds = imputer.model(X_tensor)
        medians = imputer.bar_distribution.median(logits=preds).squeeze(0)
    
    X_imputed_norm = medians.cpu().numpy()
    missing_mask = np.isnan(X_norm)
    X_norm[missing_mask] = X_imputed_norm[missing_mask]
    
    # Compute variance from distribution
    preds = preds.cpu().to(torch.float32)
    with torch.no_grad():
        p = torch.softmax(preds, dim=-1)
        bins = torch.linspace(-1, 1, preds.shape[-1])
        E_x  = (p * bins).sum(dim=-1)
        E_x2 = (p * bins**2).sum(dim=-1)
        var  = (E_x2 - E_x**2).squeeze(0).numpy()
    
    var_orig = (stds**2) * var
    uncertainty_std = np.sqrt(var_orig)
    imputed_values  = X_imputed_norm * (stds + 1e-16) + means
    
    epsilon = 1e-6
    confidence = np.clip(1.0 / (var_orig + epsilon), 0, 100)
    
    return imputed_values, confidence, uncertainty_std, var_orig
```

Note:
Confidence scores are computed from the variance of the predicted probability
distribution. No sampling required - confidence comes directly from the model's
internal distribution.

--

<!-- Vertical Slide: Evaluation -->
## Evaluating Imputation Quality

```python
# Get imputed values with confidence
imputed, confidence, uncertainty, variance = impute_with_uncertainty(imputer, X_array)

# Evaluate accuracy
original_mask = np.isnan(X_array)
mae  = np.mean(np.abs(imputed[original_mask] - X_test.values[original_mask]))
rmse = np.sqrt(np.mean((imputed[original_mask] - X_test.values[original_mask])**2))

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Mean confidence: {np.nanmean(confidence[original_mask]):.4f}")
```

### Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Confidence**: Average confidence score for imputed values

Note:
Evaluating imputation quality requires comparing imputed values against
ground truth. Confidence scores provide additional insight into prediction
uncertainty.

--

<!-- Vertical Slide: Visualization & Calibration Analysis -->
## Uncertainty Visualization & Calibration

### Four-Panel Analysis Dashboard

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.85em;">

<div style="border: 2px solid #4682B4; padding: 10px; border-radius: 5px;">

**Panel 1: Confidence Distribution**
- Histogram of confidence scores
- Shows spread of model certainty
- Mean confidence indicated

</div>

<div style="border: 2px solid #FF7F50; padding: 10px; border-radius: 5px;">

**Panel 2: Uncertainty Distribution**
- Histogram of uncertainty (std dev)
- Shows prediction variability
- Mean uncertainty indicated

</div>

<div style="border: 2px solid #2F4F2F; padding: 10px; border-radius: 5px;">

**Panel 3: Calibration Check**
- Scatter: Uncertainty vs Error
- Regression line with correlation
- Positive correlation = well-calibrated

</div>

<div style="border: 2px solid #800080; padding: 10px; border-radius: 5px;">

**Panel 4: Confidence vs Error**
- Scatter: Confidence vs Error
- Regression line with correlation
- Negative correlation expected

</div>

</div>

### Code Example

```python
import matplotlib.pyplot as plt
from scipy.stats import linregress

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Confidence Score Distribution
axes[0, 0].hist(imputed_confidence, bins=25, color='steelblue')
axes[0, 0].set_title('Distribution of Model Confidence Scores')

# 2. Uncertainty Distribution
axes[0, 1].hist(imputed_uncertainty_std, bins=25, color='coral')
axes[0, 1].set_title('Distribution of Model Uncertainties')

# 3. Calibration: Uncertainty vs Error
axes[1, 0].scatter(imputed_uncertainty_std, errors, alpha=0.5)
slope, intercept, r_value, _, _ = linregress(imputed_uncertainty_std, errors)
axes[1, 0].plot(x_line, slope * x_line + intercept, 'r--',
                label=f'r={r_value:.3f}')

# 4. Confidence vs Error
axes[1, 1].scatter(imputed_confidence, errors, alpha=0.5, color='purple')
plt.tight_layout()
plt.show()
```

Note:
These visualizations validate that the model's uncertainty estimates
are well-calibrated with actual prediction errors. See the full example
in the Imputation-SmallData.ipynb notebook.

--

<!-- Vertical Slide: Calibration Metrics -->
## Calibration Analysis

### Key Correlation Metrics

```python
from scipy.stats import linregress

# Uncertainty-Error correlation (should be positive)
slope, intercept, r_value, p_value, std_err = linregress(
    imputed_uncertainty_std, errors
)
print(f"Uncertainty-Error correlation: {r_value:.4f}")

# Confidence-Error correlation (should be negative)
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
    imputed_confidence, errors
)
print(f"Confidence-Error correlation: {r_value2:.4f}")
```

### Interpretation

| Correlation | Meaning | Quality |
|-------------|---------|---------|
| \|r\| > 0.3 | Strong correlation | ✓ Well-calibrated |
| \|r\| > 0.1 | Moderate correlation | ~ Medium calibration |
| \|r\| < 0.1 | Weak correlation | ✗ Poor calibration |

**Expected Patterns:**
- High uncertainty → High errors (positive correlation)
- High confidence → Low errors (negative correlation)

Note:
Well-calibrated models show strong correlation between predicted uncertainty
and actual errors, indicating the model "knows when it doesn't know."

--

<!-- Vertical Slide: Large Dataset Processing -->
## Large Datasets - Chunk-Based Imputation

### Row Limit: 10,000

TabPFN-based models have a **row limit of 10,000**. For larger datasets, use chunking:

```python
import numpy as np
import pandas as pd
from tabimpute.interface import ImputePFN, TabImputeCategorical

imputer = ImputePFN(device='cpu', verbose=True)
categorical = TabImputeCategorical(device='cpu')

data = pd.read_csv("large_dataset.csv")

# Identify columns with missing values
nan_cols = data[data.columns[data.isna().any()]]
col_names = data.columns[data.isna().any()]

# Separate numeric and categorical
nan_numeric_cols = nan_cols.select_dtypes(include='number')
cat_cols = nan_cols.select_dtypes(include='object').columns.tolist()

nan_cols_array_num = np.float32(np.array(nan_numeric_cols))
```

Note:
The 10,000 row limit is a fundamental constraint of TabPFN-based models.
Chunking allows processing of arbitrarily large datasets by breaking them
into manageable pieces.

--

<!-- Vertical Slide: Numeric Chunking -->
## Numeric Column Imputation (Chunked)

```python
# Numeric imputation (chunk-based)
chunks_num, _, _ = imputer.split_into_chunks(nan_cols_array_num, chunk_size=1000)

for col_idx, col in enumerate(nan_numeric_cols.columns):
    print(f"Imputing numeric column: {col}")
    imputed_col = []
    
    for i, chunk in enumerate(chunks_num):
        col_chunk = chunk[:, col_idx].reshape(-1, 1)
        imputed_chunk = imputer.impute(col_chunk)
        imputed_col.extend(imputed_chunk.flatten())
    
    nan_cols[col] = imputed_col
```

### Chunk Size Guidelines
- **Numeric**: 1000 rows per chunk
- **Memory**: Adjust based on available RAM
- **Performance**: Larger chunks = faster, but more memory

Note:
Process each column independently in chunks. The chunk size of 1000 is a
good balance between memory usage and processing speed for numeric data.

--

<!-- Vertical Slide: Categorical Chunking -->
## Categorical Column Imputation (Chunked)

```python
# Categorical imputation (chunk-based)
chunks_cat, _, _ = imputer.split_into_chunks(
    nan_cols[cat_cols].to_numpy(), 
    chunk_size=300
)

for col_idx, col in enumerate(cat_cols):
    print(f"Imputing categorical column: {col}")
    imputed_col = []
    
    for i, chunk in enumerate(chunks_cat):
        col_chunk = chunk[:, col_idx].reshape(-1, 1)
        imputed_chunk = categorical.impute(col_chunk, categorical_columns=[0])
        imputed_col.extend(imputed_chunk.flatten())
    
    nan_cols[col] = imputed_col

# Write back to original dataframe
data[col_names] = nan_cols[col_names]

# Verify
print(data.isna().sum())
data.to_csv('cleaned_data.csv', index=False)
```

**Note**: Categorical chunk size (300) is smaller due to higher memory cost

Note:
Categorical imputation requires more memory per row, so we use smaller chunks.
The chunk size of 300 is recommended for categorical data to avoid memory issues.

--

<!-- Vertical Slide: Device Support -->
## Device Support & Performance

| Device | Value  | Notes |
|--------|--------|-------|
| CPU    | `'cpu'`  | Works everywhere, slower |
| CUDA   | `'cuda'` | Recommended for large datasets on NVIDIA GPUs |
| Apple Silicon | `'mps'` | Faster on M-series Macs; requires `float32` arrays |

### Device Selection

```python
# CPU (universal)
imputer = TabImputeV2(device='cpu')

# CUDA (NVIDIA GPU)
imputer = TabImputeV2(device='cuda')

# Apple Silicon (M1/M2/M3)
imputer = TabImputeV2(device='mps')
X_array = np.array(X, dtype=np.float32)  # Required for MPS
```

Note:
Device selection impacts performance significantly. Use CUDA for NVIDIA GPUs,
MPS for Apple Silicon, and CPU as a fallback. Always cast to float32 for MPS.

--

<!-- Vertical Slide: Limitations -->
## Limitations & Considerations

### Row Limit
- **Maximum**: 10,000 rows per call
- **Solution**: Use chunk-based processing

### Data Types
- **TabImputeV2**: Numeric only
- **TabImputeCategorical**: String/object columns
- **Mixed types**: Process separately

### Requirements
- **HF Token**: Required for gated models
- **Memory**: Scales with dataset size
- **Device**: MPS requires `float32` arrays

### Confidence Scores
- Available for numeric imputation only
- Based on model's internal distribution variance
- Not available for categorical imputation

Note:
Understanding these limitations helps you design effective imputation pipelines.
Most limitations can be worked around with proper preprocessing and chunking.

--

<!-- Vertical Slide: Best Practices -->
## Best Practices

### 1. Data Preparation
```python
# Cast to float32 for MPS
X_array = np.array(X, dtype=np.float32)

# Check for infinite values
X_array = np.where(np.isinf(X_array), np.nan, X_array)
```

### 2. Chunk Size Selection
- Start with recommended sizes (1000 numeric, 300 categorical)
- Monitor memory usage
- Adjust based on available resources

### 3. Validation
```python
# Always verify no missing values remain
assert not np.isnan(X_imputed).any(), "Imputation incomplete"

# Check value ranges
assert X_imputed.min() >= expected_min
assert X_imputed.max() <= expected_max
```

### 4. Error Handling
```python
try:
    X_imputed = imputer.impute(X_array)
except Exception as e:
    print(f"Imputation failed: {e}")
    # Fallback to simpler method
```

Note:
Following these best practices ensures robust and reliable imputation pipelines
in production environments.

--

<!-- Vertical Slide: References -->
## References & Resources

### GitHub Repository
[https://github.com/jacobf18/tabular](https://github.com/jacobf18/tabular)

### Documentation
- Installation guide
- API reference
- Example notebooks

### Notebooks
| Notebook | Description |
|----------|-------------|
| `Imputation-SmallData.ipynb` | End-to-end imputation with confidence scores |
| `Imputation-HugeData.ipynb` | Chunk-based processing for large datasets |

### Related Papers
- TabPFN: Tabular Prior-Data Fitted Networks
- TabICL: Enhanced In-Context Learning for Tabular Data

Note:
The GitHub repository contains comprehensive documentation, example notebooks,
and reference implementations for various use cases.