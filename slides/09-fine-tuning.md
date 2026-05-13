<!-- Main Slide: Fine-tuning -->
## Fine-tuning Foundation Models
### Domain Adaptation Strategies

**When to Fine-tune**:
- Domain-specific patterns
- Specialized vocabulary/features
- Performance optimization
- Privacy requirements

**Approaches**:
- Full model fine-tuning
- Adapter layers
- Prompt engineering
- Ensemble with domain models

Note:
While foundation models work well zero-shot, fine-tuning can provide
additional performance gains for specific domains or applications.

--

<!-- Vertical Slide: Fine-tuning Strategies -->
## Fine-tuning Strategies

### 1. Full Model Fine-tuning
```python
from tabpfn import TabPFNClassifier

# Load pre-trained model
model = TabPFNClassifier.from_pretrained('tabpfn-v1')

# Fine-tune on domain data
model.fine_tune(
    X_domain, y_domain,
    epochs=10,
    learning_rate=1e-5,
    freeze_encoder=False  # Fine-tune all layers
)

# Use fine-tuned model
predictions = model.predict(X_test)
```

### 2. Adapter Layers (Efficient)
```python
# Add lightweight adapter layers
model = TabPFNClassifier.from_pretrained('tabpfn-v1')
model.add_adapter(
    adapter_size=64,
    adapter_type='bottleneck'
)

# Fine-tune only adapters (faster)
model.fine_tune(
    X_domain, y_domain,
    freeze_encoder=True  # Only train adapters
)
```

Note:
Adapter layers provide a parameter-efficient way to adapt the model
to new domains while preserving the general knowledge learned during
pre-training.

--

<!-- Vertical Slide: Code Example -->
## Complete Fine-tuning Example

```python
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load domain-specific data
X_domain, y_domain = load_domain_data()
X_train, X_val, y_train, y_val = train_test_split(
    X_domain, y_domain, test_size=0.2
)

# Initialize model
model = TabPFNClassifier(
    device='cuda',
    N_ensemble_configurations=16
)

# Baseline performance (zero-shot)
model.fit(X_train, y_train)
baseline_score = model.score(X_val, y_val)
print(f"Baseline accuracy: {baseline_score:.3f}")

# Fine-tuning configuration
fine_tune_config = {
    'epochs': 20,
    'learning_rate': 1e-5,
    'batch_size': 32,
    'early_stopping_patience': 5,
    'validation_split': 0.2
}

# Fine-tune model
model.fine_tune(X_train, y_train, **fine_tune_config)

# Evaluate fine-tuned model
finetuned_score = model.score(X_val, y_val)
print(f"Fine-tuned accuracy: {finetuned_score:.3f}")
print(f"Improvement: {finetuned_score - baseline_score:.3f}")

# Save fine-tuned model
model.save('models/tabpfn_finetuned_domain.pt')
```

Note:
This example shows the complete workflow from baseline evaluation
to fine-tuning and saving the adapted model.

--

<!-- Vertical Slide: Best Practices -->
## Fine-tuning Best Practices

### Data Requirements
- **Minimum**: 100-500 samples
- **Recommended**: 1000+ samples
- **Quality over quantity**: Clean, representative data

### Hyperparameters
- **Learning rate**: 1e-5 to 1e-4 (lower than pre-training)
- **Epochs**: 10-50 (use early stopping)
- **Batch size**: 16-64
- **Regularization**: Dropout 0.1-0.2

### Monitoring
- Track validation loss
- Watch for overfitting
- Compare with zero-shot baseline
- Use cross-validation

### When NOT to Fine-tune
- ❌ Very small datasets (< 100 samples)
- ❌ Zero-shot performance is sufficient
- ❌ Limited computational resources
- ❌ Frequent model updates needed

Note:
Fine-tuning requires careful consideration of trade-offs between
performance gains and computational costs. Often, zero-shot
performance is sufficient for many applications.