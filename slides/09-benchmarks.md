<!-- Main Slide: Benchmarks -->
## Performance Benchmarks
### TabPFN vs Traditional Methods

**Benchmark Datasets**:
- OpenML CC18 (72 datasets)
- UCI Machine Learning Repository
- Kaggle competitions
- Domain-specific datasets

**Comparison Methods**:
- XGBoost, LightGBM, CatBoost
- Random Forest
- Neural Networks
- AutoML systems

Note:
Comprehensive benchmarking shows that foundation models are competitive
with or superior to traditional methods on small to medium datasets,
while requiring minimal hyperparameter tuning.

--

<!-- Vertical Slide: Performance Results -->
## Benchmark Results

### OpenML CC18 (Classification)

| Method | Avg. Accuracy | Avg. Rank | Training Time |
|--------|--------------|-----------|---------------|
| **TabPFN** | **0.847** | **1.8** | **< 1s** |
| XGBoost (tuned) | 0.832 | 2.3 | 5-60s |
| LightGBM (tuned) | 0.828 | 2.5 | 3-45s |
| Random Forest | 0.815 | 3.1 | 2-30s |
| AutoML (AutoGluon) | 0.841 | 2.0 | 60-300s |

### Key Findings
- ✅ Best average performance
- ✅ Fastest inference (< 1 second)
- ✅ No hyperparameter tuning required
- ✅ Consistent across datasets

Note:
TabPFN achieves state-of-the-art results while being orders of magnitude
faster than traditional methods that require hyperparameter tuning.

--

<!-- Vertical Slide: Dataset Size Analysis -->
## Performance vs Dataset Size

### Small Datasets (< 1000 samples)
- **TabPFN**: Excellent performance
- **Traditional ML**: Often overfit or underfit
- **Deep Learning**: Poor performance

### Medium Datasets (1000-10000 samples)
- **TabPFN**: Strong performance with chunking
- **Traditional ML**: Good with tuning
- **Deep Learning**: Competitive

### Large Datasets (> 10000 samples)
- **TabPFN**: Requires chunking, performance varies
- **Traditional ML**: Excellent with resources
- **Deep Learning**: Best performance

### Recommendation
Use TabPFN for datasets < 10K samples for best results

Note:
The sweet spot for foundation models is small to medium datasets
where traditional methods struggle without extensive tuning.

--

<!-- Vertical Slide: Ablation Studies -->
## Ablation Studies

### Impact of Ensemble Size

| N_ensemble | Accuracy | Inference Time |
|------------|----------|----------------|
| 1 | 0.821 | 0.1s |
| 8 | 0.839 | 0.4s |
| 16 | 0.845 | 0.8s |
| 32 | 0.847 | 1.5s |
| 64 | 0.848 | 3.0s |

**Conclusion**: 16-32 ensemble members provide best accuracy/speed trade-off

### Impact of Training Data Size

| Training Samples | Accuracy |
|-----------------|----------|
| 50 | 0.756 |
| 100 | 0.802 |
| 250 | 0.831 |
| 500 | 0.843 |
| 1000 | 0.847 |

**Conclusion**: Performance improves with more training data, plateaus around 500-1000 samples

Note:
These ablation studies help understand the model's behavior and
guide hyperparameter selection for different scenarios.

--

<!-- Vertical Slide: Domain-Specific Results -->
## Domain-Specific Performance

### Healthcare
- **Task**: Disease prediction
- **TabPFN Accuracy**: 0.89
- **Best Traditional**: 0.86 (XGBoost)
- **Advantage**: Handles missing values naturally

### Finance
- **Task**: Credit risk assessment
- **TabPFN Accuracy**: 0.84
- **Best Traditional**: 0.85 (LightGBM)
- **Advantage**: Fast deployment, no tuning

### Manufacturing
- **Task**: Quality control
- **TabPFN Accuracy**: 0.92
- **Best Traditional**: 0.90 (Random Forest)
- **Advantage**: Real-time inference

### E-commerce
- **Task**: Customer churn prediction
- **TabPFN Accuracy**: 0.81
- **Best Traditional**: 0.82 (CatBoost)
- **Advantage**: Quick iteration

Note:
Domain-specific results show that TabPFN is competitive across
various applications, with particular strengths in scenarios
requiring fast deployment and minimal tuning.