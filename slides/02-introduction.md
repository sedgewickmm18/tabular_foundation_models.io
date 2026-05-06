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

<!-- Vertical Slide: Evolution -->
## Evolution of Tabular ML

### Traditional Approaches (2000s-2010s)
- Decision Trees, Random Forests
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks (limited success)

### AutoML Era (2015-2020)
- Automated feature engineering
- Hyperparameter optimization
- Ensemble methods

### Foundation Model Era (2020+)
- **TabPFN** (2022): Prior-fitted networks
- **TabICL** (2024+): Enhanced in-context learning
- Meta-learning on synthetic data

Note:
The evolution shows a clear trend toward automation and generalization.
Foundation models represent the culmination of this trend, offering
unprecedented flexibility and ease of use.

--

<!-- Vertical Slide: Key Papers -->
## Key Research Papers

### TabPFN (2022)
**"TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"**
- Authors: Hollmann et al.
- arXiv: 2207.01848
- Key innovation: Prior-fitted networks for tabular data

### TabICL (2024)
**"Tabular In-Context Learning"**
- Enhanced scalability
- Improved performance
- Better handling of large datasets

### Related Work
- "Attention Is All You Need" (Transformers)
- "Language Models are Few-Shot Learners" (GPT-3)
- "On the Opportunities and Risks of Foundation Models"

Note:
These papers build on the success of transformers in NLP and computer vision,
adapting the architecture and training methodology for tabular data.

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