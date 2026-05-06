<!-- Main Slide: Chunking & Reconciliation -->
## Chunking & Reconciliation
### Managing Memory and Context Windows

**Key Topics**:
- Why chunking is necessary
- Tumbling vs. sliding windows
- Mathematical foundations of reconciliation
- Implementation strategies

Note:
Before fine-tuning, we need to understand how to handle datasets that exceed
the model's context window. This section explores chunking strategies and
the mathematical techniques for reconciling predictions from multiple windows.

--

<!-- Vertical Slide: Why Chunking? -->
## Why Chunking is Necessary

### Memory Constraints in Transformers

**Transformer Memory Complexity**: $O(n^2 \cdot d)$
- $n$: sequence length (number of rows)
- $d$: model dimension
- Each attention layer requires quadratic memory

**Context Window Limitations**:
- **TabPFN**: ~1000 samples maximum
- **TabICL**: Improved efficiency, potentially higher limits
- Memory grows rapidly with dataset size

### TabICL's Advantage

TabICL's architectural improvements reduce memory consumption:
- More efficient attention mechanisms
- Better parameter sharing
- **May eliminate chunking** for moderately-sized datasets

Note:
The quadratic memory requirement of transformers means that doubling the
dataset size quadruples the memory needed. TabICL's improvements make it
more memory-efficient, but chunking is still necessary for very large datasets.

--

<!-- Vertical Slide: Tumbling Window Chunking -->
## Tumbling Window Chunking
### Static Boundaries and Their Impact

<img src="assets/images/tumbling_window_chunking.svg" alt="Tumbling Window Chunking" style="width: 50%; margin: 10px auto; display: block;">

**Concept**: Divide data into non-overlapping chunks
- Process each chunk independently
- Simple and fast
- **Problem**: Boundary placement affects imputation quality

**Key Issue**: Missing values near boundaries lack full context
- Chunk 1 context ends abruptly
- Chunk 2 context starts abruptly
- Information is lost at the boundary

Note:
The diagram shows how a static chunk boundary at position 4.5 cuts off
context. Missing values near this boundary receive lower-quality imputations
because the model cannot see the full pattern across the boundary.

--

<!-- Vertical Slide: Sliding Window Chunking -->
## Sliding Window Chunking
### Triple Overlap Strategy

<img src="assets/images/sliding_window_chunking.svg" alt="Sliding Window Chunking" style="width: 50%; margin: 10px auto; display: block;">

**Concept**: Overlapping windows ensure complete coverage
- Each missing value is seen in **3 different contexts**:
  1. At the **end** of a window
  2. In the **middle** of a window (best quality)
  3. At the **start** of a window

**Benefits**:
- No information loss at boundaries
- Multiple predictions per missing value
- Higher quality imputations

**Challenge**: How to combine multiple predictions?
→ **Reconciliation** is needed!

Note:
The sliding window approach generates multiple predictions for each missing
value. The target hole at index 5 is covered by three different windows,
each providing a different perspective. We need a principled way to combine
these predictions.

--

<!-- Vertical Slide: Mathematical Foundation - Gaussian Weighting -->
## Gaussian Weighting Function
### Distance-Based Quality Assessment

**The Weighting Formula**:

$$w_j = \exp\left(-\frac{(i - c_j)^2}{2\sigma^2}\right)$$

Where:
- $w_j$: weight for window $j$
- $i$: index of the missing value
- $c_j$: center position of window $j$
- $\sigma$: standard deviation (controls decay rate)

**Intuition**: 
- Missing values at the **center** of a window receive **highest weight**
- Quality decreases with distance from center
- Foundation models have best attention at center positions

**Why Gaussian?**
- Smooth, continuous decay
- Well-studied statistical properties
- Natural representation of attention quality

Note:
The Gaussian weighting function reflects the empirical observation that
transformer models provide the highest quality predictions for tokens
in the middle of their context window, where attention can flow freely
in both directions.

--

<!-- Vertical Slide: Linear Opinion Pool -->
## Linear Opinion Pool
### Ensemble Averaging

**Mathematical Formula**:

$$P(x_i) = \sum_{j=1}^{n} w_j \cdot P(x_i | \mathcal{C}_j)$$

Where:
- $P(x_i)$: final probability for missing value $x_i$
- $P(x_i | \mathcal{C}_j)$: probability from window $j$ with context $\mathcal{C}_j$
- $w_j$: Gaussian weight for window $j$
- $\sum_{j=1}^{n} w_j = 1$ (normalized weights)

**Characteristics**:
- **Averaging** of probability distributions
- More **permissive** of outliers
- Each window contributes proportionally to its weight
- Smooth, consensus-based predictions

**Use Case**: When you want to balance multiple uncertain predictions

Note:
The Linear Opinion Pool treats each window's prediction as a vote,
weighted by its quality. This is similar to ensemble averaging in
traditional machine learning, where multiple models vote on the outcome.

--

<!-- Vertical Slide: Log-Opinion Pool -->
## Log-Opinion Pool
### Product of Experts

**Mathematical Formula**:

$$\log P(x_i) = \sum_{j=1}^{n} w_j \cdot \log P(x_i | \mathcal{C}_j) + \text{const}$$

**Equivalent to** (after normalization):

$$P(x_i) \propto \prod_{j=1}^{n} P(x_i | \mathcal{C}_j)^{w_j}$$

**Characteristics**:
- **Product** of probability distributions (in log space)
- Acts as **intersection of beliefs**
- More **conservative**: requires agreement across windows
- Preferred for foundation models

**Why Preferred?**
- If one window is **certain** (sharp peak), result stays sharp
- If one window is **uncertain** (flat), it doesn't dominate
- Better captures the consensus of expert predictions

Note:
The Log-Opinion Pool is mathematically preferred because it treats each
window as an expert that must agree. If any window is highly confident
about a particular value, that confidence is preserved in the final result.

--

<!-- Vertical Slide: Why Log Space? -->
## Why Log Space?
### Numerical Stability and Sharpening

### 1. Numerical Stability
**Problem**: Multiplying many small probabilities
```
0.1 × 0.1 × 0.1 × ... → underflow (becomes 0)
```

**Solution**: Work in log space
```
log(0.1) + log(0.1) + log(0.1) + ... → stable
```

### 2. Sharpening Effect

**Scenario**: Two windows make predictions
- Window 1: **Certain** (sharp peak at class A)
- Window 2: **Uncertain** (flat distribution)

**Linear Pool**: Averages → flattens the sharp peak
**Log Pool**: Product → preserves the sharp peak

### 3. Intersection of Beliefs

Log-Opinion Pool finds values that **all windows agree on**:
- High probability only if **all windows** assign high probability
- Natural way to combine expert opinions
- Mathematically sound for probabilistic models

Note:
The log space transformation is not just a computational trick—it
fundamentally changes how we combine predictions. It ensures that
confident predictions are preserved and that we avoid numerical issues
with very small probabilities.

--

<!-- Vertical Slide: Implementation Logic -->
## Implementation Logic
### Step-by-Step Algorithm

**1. Windowing**
```
For each sliding window position:
    Ensure target missing value is covered
    Optimal: 3 passes (start, middle, end)
```

**2. Inference**
```
For each window j:
    Pass window through TabPFN/TabICL
    Obtain probability distribution P_j(x_i)
```

**3. Log Transformation**
```
For each window j:
    L_j = log(P_j(x_i))
    Compute Gaussian weight w_j based on distance
```

**4. Weighted Summation**
```
L_combined = Σ(w_j × L_j)
```

**5. Normalization**
```
P_final(x_i) = exp(L_combined)
P_final(x_i) = P_final(x_i) / Σ(P_final(x_i))
```

Note:
This step-by-step process ensures that we properly combine predictions
from multiple windows while maintaining numerical stability and
preserving the quality of confident predictions.

--

<!-- Vertical Slide: Pseudocode Implementation -->
## Pseudocode Implementation

```python
def reconcile_predictions(windows, missing_index, sigma=1.0):
    """
    Reconcile predictions from multiple sliding windows
    
    Args:
        windows: List of (start_pos, end_pos, predictions) tuples
        missing_index: Index of the missing value to impute
        sigma: Gaussian weight decay parameter
    
    Returns:
        Final probability distribution
    """
    log_probs = []
    weights = []
    
    # Step 1: Compute weighted log probabilities
    for start, end, pred in windows:
        # Calculate window center
        center = (start + end) / 2
        
        # Gaussian weight based on distance from center
        distance = missing_index - center
        weight = exp(-(distance**2) / (2 * sigma**2))
        
        # Transform to log space
        log_prob = log(pred + epsilon)  # Add epsilon for stability
        
        log_probs.append(log_prob)
        weights.append(weight)
    
    # Step 2: Normalize weights
    weights = weights / sum(weights)
    
    # Step 3: Weighted sum in log space (Log-Opinion Pool)
    combined_log_prob = sum(w * lp for w, lp in zip(weights, log_probs))
    
    # Step 4: Transform back to probability space
    final_prob = exp(combined_log_prob)
    
    # Step 5: Normalize to ensure sum = 1
    final_prob = final_prob / sum(final_prob)
    
    return final_prob
```

**Key Implementation Details**:
- Add small \\(\epsilon\\) (e.g., \\(10^{-10}\\)) to avoid \\(\log(0)\\)
- Normalize weights to sum to 1
- Final normalization ensures valid probability distribution

Note:
This pseudocode provides a complete implementation of the Log-Opinion Pool
reconciliation strategy. In practice, you would integrate this with your
sliding window chunking logic to handle large datasets efficiently.