<!-- Main Slide: Code Examples -->
## Practical Code Examples
### Using TabPFN in Production

**Topics Covered**:
- Basic usage and hyperparameters
- Chunking with sliding windows
- Prediction reconciliation
- Best practices

Note:
These code examples demonstrate how to use TabPFN and TabICL effectively
in real-world applications, including handling large datasets and
optimizing performance.

--

<!-- Vertical Slide: Hyperparameter Settings -->
## Hyperparameter Configuration

```python
from tabpfn import TabPFNClassifier

# Basic configuration
model = TabPFNClassifier(
    device='cpu',  # or 'cuda' for GPU
    N_ensemble_configurations=32,  # Number of ensemble members
)

# Advanced configuration
model = TabPFNClassifier(
    device='cuda',
    N_ensemble_configurations=16,  # Fewer for speed
    no_preprocess_mode=False,  # Enable preprocessing
    multiclass_decoder='permutation',  # or 'shuffle'
)

# Fit and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

Note:
The hyperparameters control the trade-off between accuracy and speed.
More ensemble configurations improve accuracy but increase inference time.

--

<!-- Vertical Slide: Chunking with Sliding Windows -->
## Handling Large Datasets: Chunking

```python
import numpy as np
from tabpfn import TabPFNClassifier

def chunk_predict(model, X_train, y_train, X_test, 
                  chunk_size=1000, overlap=100):
    """
    Predict on large datasets using sliding window chunks
    """
    n_samples = len(X_train)
    predictions = []
    
    # Create chunks with overlap
    for start in range(0, n_samples, chunk_size - overlap):
        end = min(start + chunk_size, n_samples)
        
        # Train on chunk
        X_chunk = X_train[start:end]
        y_chunk = y_train[start:end]
        
        model.fit(X_chunk, y_chunk)
        chunk_pred = model.predict_proba(X_test)
        predictions.append(chunk_pred)
    
    return predictions

# Usage
model = TabPFNClassifier()
chunk_predictions = chunk_predict(
    model, X_train, y_train, X_test,
    chunk_size=800, overlap=200
)
```

Note:
Chunking is necessary when your training dataset exceeds TabPFN's
maximum context size (~1000 samples). The overlap helps ensure
smooth transitions between chunks.

--

<!-- Vertical Slide: Reconciliation Strategies -->
## Prediction Reconciliation

```python
def reconcile_predictions(predictions, method='mean'):
    """
    Combine predictions from multiple chunks
    
    Args:
        predictions: List of probability arrays
        method: 'mean', 'median', or 'weighted'
    """
    predictions = np.array(predictions)
    
    if method == 'mean':
        # Simple average
        return np.mean(predictions, axis=0)
    
    elif method == 'median':
        # Robust to outliers
        return np.median(predictions, axis=0)
    
    elif method == 'weighted':
        # Weight by confidence (entropy)
        from scipy.stats import entropy
        weights = []
        for pred in predictions:
            # Lower entropy = higher confidence
            ent = entropy(pred.T)
            weight = 1 / (ent + 1e-10)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum(axis=0, keepdims=True)
        
        return np.sum(predictions * weights[:, np.newaxis, :], axis=0)
    
    else:
        raise ValueError(f"Unknown method: {method}")

# Usage
final_predictions = reconcile_predictions(
    chunk_predictions, 
    method='weighted'
)
```

Note:
Different reconciliation strategies work better for different scenarios.
Weighted averaging based on confidence often provides the best results.

--

<!-- Vertical Slide: Complete Pipeline -->
## Complete Production Pipeline

```python
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

class TabPFNPipeline:
    def __init__(self, chunk_size=800, overlap=200):
        self.model = TabPFNClassifier(
            device='cuda',
            N_ensemble_configurations=16
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def fit_predict(self, X_train, y_train, X_test):
        """Fit and predict with automatic chunking"""
        if len(X_train) <= self.chunk_size:
            # Small dataset: direct prediction
            self.model.fit(X_train, y_train)
            return self.model.predict_proba(X_test)
        else:
            # Large dataset: use chunking
            predictions = self._chunk_predict(
                X_train, y_train, X_test
            )
            return self._reconcile(predictions)
    
    def _chunk_predict(self, X_train, y_train, X_test):
        predictions = []
        n_samples = len(X_train)
        
        for start in range(0, n_samples, 
                          self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, n_samples)
            
            self.model.fit(
                X_train[start:end], 
                y_train[start:end]
            )
            pred = self.model.predict_proba(X_test)
            predictions.append(pred)
        
        return predictions
    
    def _reconcile(self, predictions):
        return np.mean(predictions, axis=0)

# Usage
pipeline = TabPFNPipeline(chunk_size=800, overlap=200)
probabilities = pipeline.fit_predict(X_train, y_train, X_test)
predictions = np.argmax(probabilities, axis=1)
```

Note:
This pipeline automatically handles both small and large datasets,
making it easy to use TabPFN in production without worrying about
dataset size constraints.