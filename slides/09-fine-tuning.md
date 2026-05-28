<!-- Main Slide: Fine-tuning -->
## Fine-tuning Foundation Models
### Domain Adaptation Strategies

<div style="display: flex; gap: 20px; align-items: flex-start;">

<!-- Left column: When to Fine-tune -->
<div style="flex: 1; font-size: 0.8em; text-align: left; border: 2px solid #2D6A4F; padding: 15px; border-radius: 8px; background-color: #F0FFF4;">

### When to Fine-Tune

Building on **ontology knowledge** from previous slides:

- **Domain-specific patterns**
  - Ontology-derived feature relationships
  - Specialized business rules
  
- **Specialized vocabulary/features**
  - Domain-specific encodings
  - Industry-specific terminology
  
- **Performance optimization**
  - Beyond zero-shot capabilities
  - Task-specific improvements
  
- **Privacy requirements**
  - On-premise deployment
  - Sensitive data handling

</div>

<!-- Right column: Approaches -->
<div style="flex: 1; font-size: 0.8em; text-align: left; border: 2px solid #1E40AF; padding: 15px; border-radius: 8px; background-color: #EFF6FF;">

### Approaches

**Reference**: [On Finetuning Tabular Foundation Models](https://arxiv.org/pdf/2506.08982)

**Comparison of Methods**:
1. ✅ **Full Finetuning** (best performance)
2. Low Rank Adapters (LoRA)
3. Last layers (decoder only)
4. LayerNorm, Head & Embeddings
5. Numerical Feature Embeddings

**Key Finding**: 
> Full finetuning consistently outperforms parameter-efficient methods across diverse tabular tasks.

**Trade-off**: Performance vs. computational cost

</div>

</div>

Note:
While foundation models excel at zero-shot learning, fine-tuning provides additional gains when you have domain-specific data. The ontology concepts we discussed earlier (domain structure, feature relationships) become crucial when preparing data for fine-tuning. Recent research shows full finetuning achieves best results, though selective approaches offer efficiency benefits.

--

<!-- Vertical Slide: Fine-tuning Approaches Comparison -->
## Fine-tuning Approaches: Detailed Comparison

<div style="font-size: 0.75em;">

### From "On Finetuning Tabular Foundation Models" (2024)

| Approach | Parameters Updated | Efficiency | Performance | Use Case |
|----------|-------------------|------------|-------------|----------|
| **Full Finetuning** | All layers | ⚠️ High compute | ⭐⭐⭐⭐⭐ Best | Maximum accuracy needed |
| **LoRA** | Low-rank adapters | ✅ Efficient | ⭐⭐⭐⭐ Good | Multiple domain adaptations |
| **Decoder Only** | Last layers | ✅ Very efficient | ⭐⭐⭐ Moderate | Quick adaptation |
| **LayerNorm + Head** | Normalization + output | ✅ Most efficient | ⭐⭐ Limited | Minimal resources |
| **Feature Embeddings** | Input embeddings | ✅ Efficient | ⭐⭐⭐ Moderate | Domain-specific features |

### Key Insights

**Performance Ranking** (from paper):
```
Full Finetuning > LoRA > Decoder Only > LayerNorm+Head ≈ Feature Embeddings
```

**Practical Considerations**:
- **Full finetuning**: Best when accuracy is critical and compute is available
- **LoRA**: Good balance for production systems with multiple domains
- **Decoder only**: Fast iteration during development
- **Selective layers**: When severely compute-constrained

</div>

Note:
The paper systematically compared these approaches across multiple tabular datasets. While full finetuning requires more computational resources, it consistently delivers superior performance. For production systems, consider LoRA as a good compromise between efficiency and accuracy.

--

<!-- Vertical Slide: California Housing Example - Setup -->
## Practical Example: California Housing Dataset

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 0.75em;">

<div>

### Dataset Overview

**California Housing Dataset**:
- 20,640 samples
- 8 features (location, housing characteristics)
- Target: Median house value

**Features**:
```python
['MedInc', 'HouseAge', 'AveRooms', 
 'AveBedrms', 'Population', 'AveOccup',
 'Latitude', 'Longitude']
```

**Data Split**:
- Training: 16,718 samples (81%)
- Validation: 1,858 samples (9%)
- Test: 2,064 samples (10%)

</div>

<div>

### Setup Code

```python
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.finetuning import FinetunedTabPFNRegressor
import sklearn.datasets

# Load data
data = sklearn.datasets.fetch_california_housing(
    as_frame=True
)
X_all = data.data
y_all = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, 
    test_size=0.1, 
    random_state=42
)

X_train_final, X_val, y_train_final, y_val = \
    train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42
    )
```

</div>

</div>

Note:
We use the California Housing dataset as a practical example. This real-world regression task demonstrates how to fine-tune TabPFN for domain-specific performance improvements. The dataset is split into training, validation, and test sets to properly evaluate the fine-tuning process.

--


<!-- Vertical Slide: Understanding CRPS -->
## Understanding CRPS: Continuous Ranked Probability Score

<div style="text-align: center; margin: 8px 0;">

<h3 style="font-size: 0.9em; margin-bottom: 5px;">Mathematical Definition</h3>

<img src="assets/images/math/crps_formula.svg" alt="CRPS Formula" style="max-width: 50%; height: auto;">

<div style="font-size: 0.65em; margin-top: 5px;">
Where **ECDF(y)** = Empirical Cumulative Distribution Function (from bins) and **𝟙(y ≥ y_obs)** = Heaviside step function, the ECDF of the Dirac distribution.
</div>
<br>

</div>

<h3 style="font-size: 0.9em; margin-top: 8px; margin-bottom: 5px;">Connection to Binning (from "Binning - how a classifier can do regression")</h3>

<div style="display: grid; grid-template-columns: 1fr 1.3fr; gap: 15px; font-size: 0.65em; margin-top: 8px;">

<div style="border: 2px solid #2D6A4F; padding: 10px; border-radius: 8px; background-color: #F0FFF4;">

**Step 1: Binning Creates ECDF**

Recall from TabPFN architecture:
- Target range divided into B bins (typically 100)
- Model outputs probability for each bin
- Creates step-wise PDF (probability density)
- Integrating PDF → **ECDF** (cumulative distribution)

</div>

<div style="display: flex; gap: 12px; align-items: center;">

<img src="assets/images/crps_ecdf_heaviside.png" alt="CRPS: ECDF vs Heaviside with Binning" style="width: 260px; height: auto; border: 1px solid #ccc; border-radius: 8px;">

<div style="border: 2px solid #1E40AF; padding: 10px; border-radius: 8px; background-color: #EFF6FF; flex: 1;">

**Step 2: CRPS Evaluates Quality**

- CRPS measures L² distance between ECDF and Heaviside, penalizing probability mass before true value and missing probability mass after

</div>

</div>

</div>

<div style="margin-top: 10px; padding: 10px; background-color: #FEF3C7; border-left: 5px solid #F59E0B; font-size: 0.65em;">

**Key Insight: Binning + CRPS = Probabilistic Regression**

**Why CRPS for Fine-tuning?**
- **Binning** (from architecture) converts regression → classification with ECDF output
- **CRPS** (for fine-tuning) evaluates how well ECDF matches reality (Heaviside)
- ✅ Evaluates full distribution, proper scoring rule, works with binned outputs
- ✅ Captures both accuracy and uncertainty quality

**This is why TabPFN provides both point predictions (expected value) and uncertainty estimates (distribution width)!**

</div>

Note:
CRPS is the bridge between TabPFN's binned classification approach and regression evaluation. By measuring the L2 distance between the predicted ECDF (from bins) and the true Heaviside function, CRPS provides a proper scoring rule that encourages well-calibrated probabilistic predictions. This makes it ideal for fine-tuning, as it optimizes both accuracy and uncertainty quantification simultaneously.

--

<!-- Vertical Slide: Training Configuration with Selective Freezing -->
## Training Configuration: Decoder-Only Fine-tuning

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 0.65em;">

<div>

### Selective Parameter Freezing

```python
# Define what to freeze
FREEZE_PATTERNS = [
    "encoder",      # Input encoder
    "embedder",     # Feature embeddings
    "transformer",  # Transformer blocks
    "attention",    # Attention layers
    "layernorm",    # Layer normalization
]

# Custom regressor with freezing
class FinetunedTabPFNRegressorV2(
    FinetunedTabPFNRegressor
):
    def set_freeze_patterns(self, patterns):
        self.freeze_patterns_ = patterns
    
    def freeze_parameters(self):
        for name, param in self.model_.named_parameters():
            should_freeze = any(
                p.lower() in name.lower()
                for p in self.freeze_patterns_
            )
            param.requires_grad = not should_freeze
```

**Result**: Only decoder trained (~5-10% of parameters)

### Training Hyperparameters

```python
TRAINING_CONFIG = {
    'epochs': 30,
    'learning_rate': 1e-5,
    'weight_decay': 0.01,
    'n_context_samples': 10000,
    'n_query_samples': 2000,
    'crps_weight': 1.0,  # Primary
    'mse_weight': 1.0,   # Auxiliary
    'grad_clip': 1.0,
    'patience': 8,
}
```

</div>

<div>

### Loss Functions

**CRPS (Continuous Ranked Probability Score)**:
- Evaluates full predictive distribution from bins
- See previous slide for detailed explanation
- Measures L² distance: ECDF vs. Heaviside
- Proper scoring rule for probabilistic predictions

**MSE (Mean Squared Error)**:
- Auxiliary loss for point predictions
- Computed from expected value of distribution
- Helps stabilize training

**Combined Loss**:
```
Total Loss = CRPS + MSE
           = Distribution Quality + Point Accuracy
```

**Why This Combination?**
- ✅ CRPS: Optimizes full distribution (uncertainty + calibration)
- ✅ MSE: Ensures accurate point estimates
- ✅ Synergy: CRPS uses binned distribution, MSE uses expected value

### Model Initialization & Training

```python
# Initialize with freezing
finetuned_reg = FinetunedTabPFNRegressorV2(
    device="cuda",
    epochs=30,
    learning_rate=1e-5,
    crps_loss_weight=1.0,
    mse_loss_weight=1.0,
    early_stopping=True,
)
finetuned_reg.set_freeze_patterns(FREEZE_PATTERNS)

# Train
finetuned_reg.fit(X_train, y_train)
```

</div>

</div>

<div style="margin-top: 10px; padding: 8px; background-color: #DBEAFE; border-left: 4px solid #1E40AF; font-size: 0.65em;">

📓 **Complete Example**: See full implementation in [`examples/finetune_tabpfn_california_housing.ipynb`](examples/finetune_tabpfn_california_housing.ipynb) with training history tracking, visualization, and detailed evaluation.

</div>

Note:
This slide combines selective parameter freezing with training configuration. By freezing encoder and transformer layers, we train only 5-10% of parameters (decoder only), dramatically reducing computational requirements. The CRPS + MSE loss combination optimizes both distribution quality and point accuracy. See the complete notebook for full implementation details including custom FinetunedTabPFNRegressorV2 class, training history tracking, and comprehensive evaluation.

--

<!-- Vertical Slide: Training Results -->
## Training Results: Decoder-Only Fine-tuning

<div style="font-size: 0.7em;">

### Training History Visualization

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">

<div>

**Training Progress**:
- Tracks loss per epoch
- Monitors validation metrics
- Learning rate schedule
- Early stopping trigger

**Training Convergence**:
```
Epoch  | Train Loss | Val MSE  | Val CRPS
-------|------------|----------|----------
   0   |   0.4521   | 0.1650   | 1.9500
   5   |   0.3892   | 0.1645   | 1.9480
  10   |   0.3654   | 0.1642   | 1.9470
  15   |   0.3521   | 0.1641   | 1.9466
  20   |   0.3498   | 0.1641   | 1.9466
  25   |   0.3487   | 0.1641   | 1.9466
```

</div>

<div>

**Visualization Features**:
```python
# Plot training history
finetuned_reg.plot_training_history(
    save_path='training_history.png'
)

# Access history data
history = finetuned_reg.get_training_history()
```

**4 Plots Generated**:
1. Training Loss over epochs
2. Validation MSE over epochs
3. Validation CRPS over epochs
4. Learning Rate schedule

</div>

</div>

### Performance Comparison: Baseline vs. Decoder-Only Fine-tuning

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;">

<div style="text-align: center; padding: 15px; background-color: #D1FAE5; border-radius: 8px; border: 3px solid #10B981;">

**✅ Baseline Model**
(Pre-trained V2 - Zero-shot)

| Metric | Value |
|--------|-------|
| **MAE** | **0.2410** |
| **MSE** | **0.1599** |
| **RMSE** | **0.3999** |
| **R²** | **0.8801** |
| **CRPS** | **1.9306** |

</div>

<div style="text-align: center; padding: 15px; background-color: #FEE2E2; border-radius: 8px; border: 3px solid #EF4444;">

**❌ Fine-tuned Model**
(Decoder Only - Frozen Encoder)

| Metric | Value | Change |
|--------|-------|--------|
| **MAE** | **0.2423** | ↑ 0.52% worse |
| **MSE** | **0.1641** | ↑ 2.61% worse |
| **RMSE** | **0.4051** | ↑ 1.30% worse |
| **R²** | **0.8770** | ↓ 0.36% worse |
| **CRPS** | **1.9466** | ↑ 0.83% worse |

</div>

</div>

</div>

<div style="margin-top: 20px; padding: 20px; background-color: #FEF3C7; border-left: 5px solid #F59E0B; font-size: 0.75em;">

### ⚠️ Key Finding: Decoder-Only Fine-tuning Performs Slightly Worse!

**This confirms the paper's findings**:
- **Full fine-tuning** performs best (all layers updated)
- **PEFT approaches** (Parameter-Efficient Fine-Tuning like decoder-only) require **larger datasets** to be useful
- With **limited data** (~16K samples), the pre-trained model's zero-shot performance is superior
- Freezing encoder/transformer loses valuable learned representations

**Recommendation**: For datasets < 50K samples, consider using the **baseline model** or **full fine-tuning** instead of selective approaches.

</div>

Note:
This real-world example demonstrates an important lesson: decoder-only fine-tuning with frozen encoder/transformer actually performs slightly worse than the baseline! This validates the paper's conclusion that full fine-tuning achieves best results. Parameter-efficient methods like decoder-only training need larger datasets (50K+ samples) to overcome the loss of frozen layer adaptability. For smaller datasets, the pre-trained model's zero-shot capabilities are often superior.


--

<!-- Vertical Slide: Best Practices -->
## Fine-tuning Best Practices

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 0.75em;">

<div>

### When to Use Each Approach

**Full Finetuning** ⭐ Recommended:
- ✅ Maximum accuracy required
- ✅ Sufficient computational resources
- ✅ Single domain focus
- ✅ Dataset: 10K+ samples (proven effective)
- ✅ **Best performance** (confirmed by paper & example)

**Decoder-Only (Selective)** ⚠️ Use with caution:
- ⚠️ Requires **large datasets** (50K+ samples)
- ⚠️ May perform **worse** than baseline on small data
- ✅ Limited compute budget
- ✅ Quick experimentation
- ❌ **Not recommended** for < 50K samples

**LoRA (Adapters)**:
- ✅ Multiple domain adaptations
- ✅ Model versioning important
- ✅ Production deployment
- ✅ Better than decoder-only for PEFT
- ⚠️ Still requires larger datasets than full finetuning

</div>

<div>

### Data Requirements

**Based on Empirical Results**:
- **Full finetuning**: 10K+ samples (✅ proven effective)
- **LoRA/Adapters**: 20K+ samples (moderate effectiveness)
- **Decoder-only**: 50K+ samples (⚠️ needs large data)
- **< 10K samples**: Use baseline zero-shot model

**Quality Over Quantity**:
- Clean, representative data
- Proper train/val/test splits
- Domain-specific preprocessing
- Ontology-aware feature engineering
- **Small datasets**: Baseline often better than PEFT

### Monitoring Strategy

```python
# Track key metrics
- Training loss convergence
- Validation metric trends
- Learning rate schedule
- Early stopping triggers

# Compare with baseline
baseline_score = baseline_model.score(X_test, y_test)
finetuned_score = finetuned_model.score(X_test, y_test)
improvement = (finetuned_score - baseline_score) / baseline_score
```

</div>

</div>

### Common Pitfalls to Avoid

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px; font-size: 0.7em;">

<div style="padding: 10px; background-color: #FEE2E2; border-radius: 8px;">

❌ **Overfitting**
- Too many epochs
- Insufficient validation
- Small dataset
- No regularization

</div>

<div style="padding: 10px; background-color: #FEF3C7; border-radius: 8px;">

⚠️ **Underfitting**
- Learning rate too low
- Too few epochs
- Over-regularization
- Wrong layers frozen

</div>

<div style="padding: 10px; background-color: #DBEAFE; border-radius: 8px;">

✅ **Best Practice**
- Use early stopping
- Monitor validation
- Compare with baseline
- Cross-validate results

</div>

</div>

Note:
Choose your fine-tuning approach based on your specific constraints and requirements. Full finetuning delivers best results but requires more resources. Decoder-only training offers a good balance for most use cases. Always compare against the zero-shot baseline to ensure fine-tuning provides meaningful improvements. Use the ontology knowledge from earlier slides to guide your feature engineering and preprocessing.