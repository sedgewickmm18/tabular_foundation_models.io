<!-- Main Slide: Introduction -->
## What are Foundation Models for Tables?

### The Paradigm Shift
Traditional ML for tabular data:
- Train a model for each specific task
- Extensive hyperparameter tuning
- Feature engineering required
- Limited generalization

**Foundation Models**:
- Single pre-trained model
- Zero-shot or few-shot learning
- Minimal tuning required
- Generalizes across tasks

Note:
Foundation models represent a fundamental shift in how we approach machine learning
for tabular data. Instead of training task-specific models, we use a single
pre-trained model that can adapt to new tasks through in-context learning.

--

<!-- Vertical Slide: Evolution & Key Papers -->
## Evolution & Key Research

<div style="display: flex; gap: 2rem; align-items: flex-start;">

<div style="flex: 1; border: 2px solid #666; padding: 1.5rem; border-radius: 8px; text-align: left;">

### Evolution of Tabular ML

**2000s-2010s**: Decision Trees, Gradient Boosting (XGBoost, LightGBM)

**2015-2020**: AutoML with automated feature engineering & hyperparameter tuning

**2020+**: Foundation Models - **TabPFN** (2022), **TabICL** (2024+)

</div>

<div style="flex: 1; border: 2px solid #666; padding: 1.5rem; border-radius: 8px; text-align: left;">

### Key Research Papers

**TabPFN (2022)** - Hollmann et al. (arXiv: 2207.01848)
Prior-fitted networks for small tabular classification

**TabICL (2024)** - Enhanced scalability & large dataset handling

**Foundation**: Transformers (Vaswani et al.), GPT-3 few-shot learning

</div>

</div>

Note:
Clear trend toward automation and generalization. Foundation models adapt transformer success from NLP/vision to tabular data through meta-learning on synthetic data.

--

<!-- Vertical Slide: Comparison -->
## Foundation Models vs Traditional ML

| Aspect | Traditional ML | Foundation Models |
|--------|---------------|-------------------|
| **Training** | Task-specific | Meta-learning |
| **Adaptation** | Retrain from scratch | In-context learning |
| **Data Requirements** | Large datasets | Small datasets OK |
| **Hyperparameters** | Extensive tuning | Minimal tuning |
| **Inference Speed** | Varies | Fast (~seconds) |
| **Generalization** | Limited | Broad |
| **Feature Engineering** | Often required | Automatic |

Note:
The comparison highlights the key advantages of foundation models.
However, traditional methods still have their place, especially for
very large datasets or when computational resources are limited.