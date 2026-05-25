<!-- Main Slide: Appendix B -->
## Appendix B: Chunking & Reconciliation
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

<!-- Vertical Slide: Mathematical Background -->
## Predictive posterior by example with sliding-window linear SCMs

<div style="border: 2px solid #ccc; padding: 15px; margin-bottom: 20px; text-align: left; font-size: 0.65em; line-height: 1.2;">

The posterior predictive distribution (PPD) gives the probability of a new target y<sub>new</sub> given features x<sub>new</sub> and training data *D*. It marginalizes over tasks &Phi; generated from SCM DAGs:

<img src="assets/images/math/posterior_predictive_distribution.svg" alt="Posterior Predictive Distribution" style="display: block; margin: 0.5em auto 0.5em auto; max-width: 30%;">

Tasks producing similar training data (e.g., near-linear relationships) have higher likelihood P(D|&Phi;), better approximating the integral.

</div>

<div class="columns">
<div class="column" style="flex: 0 0 70%;">

<div class="plot-container compact">
  <iframe src="assets/interactive/marginal_effects.html"
          title="Marginal Effects Analysis"
          style="width: 100%; height: 100%; border: none;">
    <img src="assets/images/marginal_effects.png" alt="Marginal Effects">
  </iframe>
</div>

</div>
<div class="column" style="flex: 0 0 30%; border: 2px solid #ccc; padding: 10px; text-align: left; font-size: 0.6em; line-height: 1.2; display: flex; align-items: flex-end; padding-bottom: 30px;margin-top: 5%">

<div>

This example uses a localized weight matrix for linear transformations on bivariate tabular data without bias, affecting only rows with distance &le; 5. Coeffients are sampled from ]-1,1[.

This is a far cry from the generic SCM process described on the previous slide, but highlights the structural impacts.

In the second figure (right) coeffiencts are further restricted to positive numbers.

</div>

</div>
</div>

<div style="border: 2px solid #ccc; padding: 15px; margin-bottom: 20px; text-align: left; font-size: 0.65em; line-height: 1.2;">

**The "Fan" Spread**: The teal plot shows a strictly positive "causal cone." This visualizes a prior where the foundation model assumes that increasing an input never results in a decrease in the output across the marginalized window. Starting with positive values in the first 5 columns results in positive values 'forever'.

**Marginalization Effect**: By summing the weights in the code (np.sum(weights)), we are effectively integrating out the row-specific variations to see the total structural impact over your 5-row lookback.

**Density vs. Bound**: The shaded areas represent the functional uncertainty. In the mixed case, the uncertainty is symmetric around zero; in the positive case, the uncertainty is "pushed" into the upper quadrant, representing a strong inductive bias.

</div>

Note:
This example demonstrates how the posterior predictive distribution works in practice with sliding-window linear SCMs. The visualization shows how different structural causal models with varying coefficients create different prediction patterns, and how the marginalization process integrates over these possibilities weighted by their likelihood given the training data.

--

<!-- Vertical Slide: Why Chunking? -->
## Why Chunking is Necessary

### Memory Constraints in Transformers

**Transformer Memory Complexity**: <img src="assets/images/math/transformer_memory.svg" alt="O(n^2 * d)" style="display: inline; vertical-align: middle; height: 1.2em; margin: 0 0.2em;">
- *n*: sequence length (number of rows)
- *d*: model dimension
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

<!-- Vertical Slide: Mathematical Foundation - Gaussian Weighting -->
## Gaussian Weighting Function
### Distance-Based Quality Assessment

<div style="text-align: center; margin-bottom: 1em;">
<img src="assets/images/math/gaussian_weight.svg" alt="Gaussian weight formula" style="max-width: 60%;">
</div>

<div style="display: flex; gap: 20px; align-items: flex-start;">

<div style="flex: 1.2;">

<div class="plot-container compact">
  <iframe src="assets/interactive/gaussian_decay_plot.html"
          title="Gaussian Decay Visualization"
          style="width: 100%; height: 100%; border: none;">
    <img src="assets/images/gaussian_decay_plot.png" alt="Gaussian decay visualization">
  </iframe>
</div>

<div style="font-size: 0.75em; text-align: center; margin-top: 0.5em;">
<strong>Where:</strong> <em>w<sub>j</sub></em>: weight for window <em>j</em>,
<em>i</em>: index of the missing value,
<em>c<sub>j</sub></em>: center position of window <em>j</em>,
<em>σ</em>: standard deviation (controls decay rate)
</div>

</div>

<div style="flex: 0.8; border: 2px solid #2D6A4F; padding: 15px; border-radius: 8px; background-color: #F0FFF4; font-size: 0.85em;">

**Intuition**:
- Missing values at the **center** of a window receive **highest weight**
- Quality decreases with distance from center
- Foundation models have best attention at center positions

**Why Gaussian?**
- Smooth, continuous decay
- Well-studied statistical properties
- Natural representation of attention quality

</div>

</div>

Note:
The Gaussian weighting function reflects the empirical observation that
transformer models provide the highest quality predictions for tokens
in the middle of their context window, where attention can flow freely
in both directions.

--

<!-- Vertical Slide: Linear Opinion Pool -->
## Linear Opinion Pool
### Ensemble Averaging

<div style="text-align: center; margin-bottom: 1em;">
<img src="assets/images/math/linear_opinion_pool.svg" alt="Linear Opinion Pool formula" style="max-width: 50%;">
</div>

<div style="display: flex; gap: 20px; align-items: flex-start;">

<div style="flex: 1.2; border: 2px solid #FF8C00; padding: 15px; border-radius: 8px; background-color: #FFF8E7;">
<img src="assets/images/linear_opinion_pooling_plot.png" alt="Linear Opinion Pool visualization" style="display: block; margin: 0 auto; max-width: 85%;">
</div>

<div style="flex: 0.8; border: 2px solid #2D6A4F; padding: 15px; border-radius: 8px; background-color: #F0FFF4; font-size: 0.85em;">

**Where**:
- *P(x<sub>i</sub>)*: final probability
- *P(x<sub>i</sub> | C<sub>j</sub>)*: probability from window *j*
- *w<sub>j</sub>*: Gaussian weight
- Σ*w<sub>j</sub>* = 1 (normalized)

**Characteristics**:
- **Averaging** of probability distributions
- More **permissive** of outliers
- Each window contributes proportionally to its weight
- Smooth, consensus-based predictions

**Use Case**: Balance multiple uncertain predictions

</div>

</div>

Note:
The Linear Opinion Pool treats each window's prediction as a vote,
weighted by its quality. This is similar to ensemble averaging in
traditional machine learning, where multiple models vote on the outcome.

--

<!-- Vertical Slide: Why Log Space? -->
## Why Log Space?
### Numerical Stability and Sharpening

<div style="display: flex; gap: 12px; align-items: stretch; margin-bottom: 0.8em;">

<div style="flex: 1; border: 2px solid #1E88E5; padding: 10px; border-radius: 8px; background-color: #E3F2FD; font-size: 0.7em;">

**1. Numerical Stability**

**Problem**: Multiplying small probabilities
```
0.1 × 0.1 × ... → 0
```

**Solution**: Log space
```
log(0.1) + ... → stable
```

Prevents underflow.

</div>

<div style="flex: 1; border: 2px solid #7B1FA2; padding: 10px; border-radius: 8px; background-color: #F3E5F5; font-size: 0.7em;">

**2. Sharpening Effect**

Two windows:
- Window 1: **Certain**
- Window 2: **Uncertain**

**Linear**: Flattens peak

**Log**: Preserves peak

Confidence preserved!

</div>

<div style="flex: 1; border: 2px solid #388E3C; padding: 10px; border-radius: 8px; background-color: #E8F5E9; font-size: 0.7em;">

**3. Intersection of Beliefs**

Finds agreement:

- High prob only if **all** agree
- Natural expert combination
- Mathematically sound

Agreement required!

</div>

</div>

<div style="text-align: center;">
<img src="assets/images/log_space_comparison.png" alt="Log space comparison" style="max-width: 75%;">
</div>

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
- Add small <img src="assets/images/math/epsilon.svg" alt="epsilon" style="display: inline; vertical-align: middle; height: 1em; margin: 0 0.1em;"> (e.g., <img src="assets/images/math/ten_to_minus_ten.svg" alt="10^-10" style="display: inline; vertical-align: middle; height: 1em; margin: 0 0.1em;">) to avoid <img src="assets/images/math/log_zero.svg" alt="log(0)" style="display: inline; vertical-align: middle; height: 1em; margin: 0 0.1em;">
- Normalize weights to sum to 1
- Final normalization ensures valid probability distribution

Note:
This pseudocode provides a complete implementation of the Log-Opinion Pool
reconciliation strategy. In practice, you would integrate this with your
sliding window chunking logic to handle large datasets efficiently.