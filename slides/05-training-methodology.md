<!-- Main Slide: Training Methodology -->
## Training Methodology
### Meta-Learning on Synthetic Data

**Core Approach**:
- Generate diverse synthetic datasets
- Train transformer to predict on any dataset
- Learn prior distribution over datasets
- Enable zero-shot generalization

Note:
The training methodology is what enables foundation models to work
without task-specific training. By learning from diverse synthetic
datasets, the model learns general patterns that transfer to real data.

--

<!-- Vertical Slide: Synthetic Data Generation -->
## Synthetic Data Generation

### Dataset Properties
- Number of samples: 10-1000
- Number of features: 1-100
- Feature types: numerical, categorical
- Target complexity: linear to highly non-linear

### Generation Process
1. Sample dataset configuration
2. Generate feature distributions
3. Create target relationships
4. Add noise and correlations
5. Introduce missing values

Note:
The diversity of synthetic datasets is crucial for generalization.
The model must see a wide variety of patterns during training.

--

<!-- Vertical Slide: Meta-Learning Objective -->
## Meta-Learning Objective

### Training Loss
$$\mathcal{L} = \mathbb{E}_{D \sim p(D)} \left[ -\log P(y_{test}|x_{test}, D_{train}, \theta) \right]$$

Where:
- $D$: synthetic dataset
- $D_{train}$: training portion
- $(x_{test}, y_{test})$: test examples
- $\theta$: model parameters

### Goal
Learn $\theta$ that minimizes expected loss across all possible datasets

Note:
The meta-learning objective ensures the model learns to adapt to
any dataset by observing training examples in context.

--

<!-- Vertical Slide: Hyperparameters -->
## Training Hyperparameters

### Model Architecture
- Transformer layers: 6-12
- Attention heads: 8-16
- Hidden dimension: 512-1024
- Dropout: 0.1-0.2

### Training Configuration
- Batch size: 32-128 datasets
- Learning rate: 1e-4 to 1e-3
- Optimizer: AdamW
- Training steps: 100K-1M

### Synthetic Data Parameters
- Dataset diversity: High
- Complexity range: Simple to complex
- Noise levels: 0-30%

Note:
These hyperparameters are carefully tuned to balance model capacity,
training efficiency, and generalization performance.