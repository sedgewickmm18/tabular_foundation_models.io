<!-- Main Slide: Use Cases Overview -->
## Foundation Models for Tables
### Key Use Cases

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem;">

<div>

### 🧩 Missing Data Imputation
- Handle incomplete datasets
- Preserve multivariate relationships
- Uncertainty quantification

</div>

<div>

### 🔍 Anomaly Detection
- Identify outliers and unusual patterns
- Zero-shot detection capability
- Multivariate analysis

</div>

</div>

#### Why Foundation Models?
- ✅ Zero-shot or few-shot learning
- ✅ No extensive hyperparameter tuning
- ✅ Competitive with task-specific models
- ✅ Fast inference (seconds)

Note:
Foundation models for tabular data represent a paradigm shift in how we approach common ML tasks.
Unlike traditional methods that require extensive tuning and training, these models leverage
in-context learning to adapt to new datasets instantly.

--

<!-- Vertical Slide 1: Deep Dive into Imputation -->
## Missing Data Imputation
### Deep Dive

#### Traditional Approaches
- **Simple**: Mean/Median imputation
- **Statistical**: K-Nearest Neighbors (KNN)
- **Advanced**: Multiple Imputation by Chained Equations (MICE)
- **Matrix-based**: Matrix factorization methods

#### Foundation Model Approach
- **In-context learning**: Model learns patterns from observed data
- **Probabilistic predictions**: Uncertainty quantification
- **Multivariate dependencies**: Captures complex relationships
- **No training required**: Zero-shot imputation

Note:
Traditional imputation methods often make strong assumptions about the data distribution
or require careful feature engineering. Foundation models, in contrast, can adapt to
the specific patterns in your data through in-context learning.

--

<!-- Vertical Slide 1.1: Missing Data Patterns -->
### Missing Data Patterns

#### 1. MCAR (Missing Completely At Random)
- Missing values independent of data
- **Example**: Random sensor failures
- Easiest to handle

#### 2. MAR (Missing At Random)
- Missing values depend on observed data
- **Example**: Income missing for certain age groups
- Requires modeling observed relationships

#### 3. MNAR (Missing Not At Random)
- Missing values depend on unobserved data
- **Example**: High earners not reporting income
- Most challenging scenario

Note:
Understanding the missing data mechanism is crucial for choosing the right imputation strategy.
Foundation models can handle all three patterns, but performance varies depending on the
complexity of the underlying mechanism.

--

<!-- Vertical Slide 1.2: Mathematical Formulation -->
### Mathematical Formulation

Given dataset with missing values:

<img src="assets/images/math/missing_data_matrix.svg" alt="X = [X_obs, X_miss]" style="display: block; margin: 0.5em auto; max-width: 60%;">

**Goal**: Estimate the conditional distribution

<img src="assets/images/math/missing_data_goal.svg" alt="P(X_miss | X_obs, theta)" style="display: block; margin: 0.5em auto; max-width: 60%;">

**Foundation Model Approach**:

<img src="assets/images/math/missing_data_approach.svg" alt="P(X_miss | X_obs) approx f_theta(X_obs)" style="display: block; margin: 0.5em auto; max-width: 70%;">

where $\theta$ is learned from meta-training on diverse synthetic datasets

**Key Advantage**: No dataset-specific training required!

Note:
The mathematical foundation of these models is based on Bayesian inference.
The model learns a prior distribution over datasets during meta-training,
which it then uses to make predictions on new data through in-context learning.

--

<!-- Vertical Slide 1.3: Code Example for Imputation -->
### Code Example: Imputation

```python
from tabpfn import TabPFNRegressor
import numpy as np

# Load data with missing values (NaN)
X_train = np.array([
    [1.0, 2.0, np.nan],
    [2.0, np.nan, 4.0],
    [3.0, 4.0, 5.0],
    [np.nan, 5.0, 6.0]
])
y_train = np.array([10, 15, 20, 25])

# TabPFN handles missing values internally
model = TabPFNRegressor()
model.fit(X_train, y_train)

# Predict on new data with missing values
X_test = np.array([
    [1.5, np.nan, 3.5],
    [np.nan, 3.0, 4.5]
])
predictions = model.predict(X_test)

print(f"Predictions: {predictions}")
```

**No explicit imputation step needed!**

Note:
One of the key advantages of TabPFN is that it handles missing values natively.
You don't need to preprocess your data or choose an imputation strategy.
The model learns to handle missingness through its training on diverse synthetic datasets.

--

<!-- Vertical Slide 1.4: Real-World Imputation Examples -->
### Real-World Applications

#### 🏥 Healthcare
- **Challenge**: Missing lab results in patient records
- **Solution**: Impute based on patient demographics and available tests
- **Benefit**: Improved diagnostic accuracy

#### 💰 Finance
- **Challenge**: Incomplete transaction data
- **Solution**: Fill gaps using temporal and customer patterns
- **Benefit**: Better fraud detection and risk assessment

#### 📡 IoT & Sensors
- **Challenge**: Sensor data with gaps due to failures
- **Solution**: Impute using correlated sensor readings
- **Benefit**: Continuous monitoring and alerting

#### 📊 Survey Data
- **Challenge**: Non-responses in questionnaires
- **Solution**: Impute based on respondent characteristics
- **Benefit**: More complete analysis and insights

Note:
These real-world examples demonstrate the versatility of foundation models.
The same model can be applied across different domains without modification,
adapting to the specific patterns in each dataset through in-context learning.

--

<!-- Vertical Slide 2: Deep Dive into Anomaly Detection -->
## Anomaly Detection
### Deep Dive

#### What is Anomaly Detection?
Identifying data points that deviate significantly from normal patterns

#### Traditional Approaches
- **Statistical**: Z-score, IQR (Interquartile Range)
- **Distance-Based**: LOF (Local Outlier Factor), k-NN
- **Density-Based**: DBSCAN
- **Isolation-Based**: Isolation Forest
- **Ensemble**: Multiple detectors combined

#### Foundation Model Approach
- **Likelihood-based**: Low probability = anomaly
- **In-context learning**: Adapts to data distribution
- **Multivariate**: Considers feature interactions
- **Zero-shot**: No training required

Note:
Traditional anomaly detection methods often struggle with high-dimensional data
and complex feature interactions. Foundation models naturally handle these challenges
through their transformer-based architecture and meta-learning approach.

--

<!-- Vertical Slide 2.1: Types of Anomalies -->
### Types of Anomalies

#### 1. Point Anomalies
- Individual data points that are anomalous
- **Example**: Single fraudulent transaction
- Most common type

#### 2. Contextual Anomalies
- Anomalous in specific context, normal otherwise
- **Example**: High temperature in winter
- Requires understanding of context

#### 3. Collective Anomalies
- Group of points anomalous together
- **Example**: Coordinated attack pattern
- Most challenging to detect

Note:
Foundation models excel at detecting all three types of anomalies because they
can capture complex patterns and relationships in the data. The in-context learning
mechanism allows them to understand what's "normal" for a specific dataset.

--

<!-- Vertical Slide 2.2: Mathematical Formulation -->
### Anomaly Detection: Math

**Anomaly Score** based on likelihood:

<img src="assets/images/math/anomaly_score.svg" alt="s(x) = -log P(x | X_context, theta)" style="display: block; margin: 0.5em auto; max-width: 70%;">

Where:
- *x*: test instance
- *X<sub>context</sub>*: normal examples (context)
- *θ*: foundation model parameters

**Decision Rule**:

<img src="assets/images/math/anomaly_decision.svg" alt="x is anomaly if s(x) > tau" style="display: block; margin: 0.5em auto; max-width: 60%;">

where *τ* is a threshold (can be learned or set based on desired false positive rate)

**Advantages**:
- Probabilistic interpretation
- Natural uncertainty quantification
- Handles multivariate dependencies

Note:
The likelihood-based approach provides a principled way to detect anomalies.
Unlike distance-based methods, it naturally accounts for the correlation structure
in the data and provides interpretable scores.

--

<!-- Vertical Slide 2.3: Detection Strategies -->
### Detection Strategies

#### 1. Supervised Anomaly Detection
```python
from tabpfn import TabPFNClassifier

# Train with labeled normal/anomaly data
X_train, y_train = load_labeled_data()  
# y: 0=normal, 1=anomaly

model = TabPFNClassifier()
model.fit(X_train, y_train)

# Predict on new data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

**Use when**: You have labeled anomalies

Note:
Supervised detection is the most straightforward approach when you have
labeled examples of both normal and anomalous instances.

--

<!-- Vertical Slide 2.4: Semi-supervised Detection -->
### Semi-supervised Detection

```python
# Train only on normal data
X_normal = load_normal_data()
y_normal = np.zeros(len(X_normal))  # All labeled as normal

model = TabPFNClassifier()
model.fit(X_normal, y_normal)

# Low probability of "normal" class = anomaly
X_test = load_test_data()
normal_probs = model.predict_proba(X_test)[:, 0]
anomaly_scores = 1 - normal_probs

# Detect anomalies
threshold = 0.5  # or use percentile
anomalies = X_test[anomaly_scores > threshold]
```

**Use when**: You only have normal examples

Note:
Semi-supervised detection is common in practice because anomalies are often rare
and difficult to collect. This approach trains the model to recognize what's normal,
and anything that doesn't fit that pattern is flagged as anomalous.

--

<!-- Vertical Slide 2.5: Real-World Use Cases -->
### Real-World Use Cases

#### 💳 Fraud Detection
- Credit card transactions
- Insurance claims
- Identity verification
- **Benefit**: Real-time fraud prevention

#### 🏭 Quality Control
- Manufacturing defects
- Product inspection
- Process monitoring
- **Benefit**: Reduced waste and costs

#### 🖥️ System Monitoring
- Network intrusion detection
- Server performance anomalies
- Application errors
- **Benefit**: Proactive issue resolution

#### 🏥 Healthcare
- Disease outbreak detection
- Unusual patient vitals
- Medical imaging anomalies
- **Benefit**: Early intervention

Note:
These use cases demonstrate the broad applicability of foundation models for
anomaly detection. The same model architecture can be applied across different
domains, adapting to the specific characteristics of each application.

--

<!-- Vertical Slide 2.6: Performance Comparison -->
### Foundation Models vs Traditional Methods

| Aspect | Traditional Methods | Foundation Models |
|--------|-------------------|-------------------|
| **Training** | Extensive tuning required | Zero/few-shot capable |
| **Adaptability** | Fixed to training distribution | Adapts via in-context learning |
| **Multivariate** | Often univariate or simple | Natural multivariate handling |
| **Uncertainty** | Limited or none | Probabilistic predictions |
| **Speed** | Varies widely | Fast inference (~seconds) |
| **Interpretability** | Method-dependent | Probability-based scores |

#### When Foundation Models Excel
- ✅ Small to medium datasets (< 10K samples)
- ✅ Mixed feature types
- ✅ Need for quick deployment
- ✅ Limited labeled anomalies

Note:
While foundation models offer many advantages, they're not always the best choice.
For very large datasets or when you have abundant labeled data and computational
resources, traditional methods or deep learning approaches might be more suitable.

--

<!-- Vertical Slide 2.7: Complete Pipeline Example -->
### Complete Anomaly Detection Pipeline

```python
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Load data
X_train, y_train = load_training_data()  # Normal + some anomalies
X_test, y_test = load_test_data()

# Train model
model = TabPFNClassifier()
model.fit(X_train, y_train)

# Predict anomaly probabilities
anomaly_probs = model.predict_proba(X_test)[:, 1]

# Evaluate
auc_score = roc_auc_score(y_test, anomaly_probs)
print(f"AUC-ROC: {auc_score:.3f}")

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(
    y_test, anomaly_probs
)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Detect anomalies with optimal threshold
predictions = (anomaly_probs > optimal_threshold).astype(int)
print(f"Detected {predictions.sum()} anomalies")
```

Note:
This complete example shows a production-ready pipeline including model training,
evaluation, threshold optimization, and final predictions. The threshold selection
is crucial and should be based on your specific requirements for precision vs recall.