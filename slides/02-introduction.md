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

<!-- Vertical Slide: On Tabular and Time Series Data -->
## On Tabular and Time Series Data

<div style="text-align: center;">
<img src="assets/images/fft_duality_diagram.png" alt="FFT Duality Diagram" style="max-width: 70%; height: auto;">
</div>

<div style="margin-top: 1.5rem; padding: 1rem; background-color: #f8f9fa; border: 2px solid #3498db; border-radius: 8px; font-size: 0.9em;">

Since tabular and time series data can be regarded as two sides of the same coin, the architecture foundation models for tabular and time series data also share common elements.

</div>

Note:
This diagram illustrates the fundamental duality between tabular data and time series through FFT/IFFT transformations. Each row in a table can be viewed as frequency weights (volume knobs) that synthesize a time series signal via IFFT. Conversely, a time series window can be analyzed via FFT to extract frequency components that form a tabular row. This mathematical equivalence explains why transformer architectures work well for both domains.

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