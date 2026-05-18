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

<img src="assets/images/imputation_explanation.png" alt="Imputation Process" style="width: 90%; max-width: 1200px; margin: 0.5rem auto; display: block;">

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1rem; font-size: 0.75em;">

<div style="border: 2px solid #E74C3C; padding: 1rem; border-radius: 8px; background-color: #FADBD8;">

#### Traditional Approaches
- **Simple**: Mean/Median imputation
- **Statistical**: K-Nearest Neighbors (KNN)
- **Advanced**: MICE
- **Matrix-based**: Matrix factorization

</div>

<div style="border: 2px solid #27AE60; padding: 1rem; border-radius: 8px; background-color: #E8F8F5;">

#### Foundation Model Approach
- **In-context learning**: Learns from observed data
- **Probabilistic predictions**: Uncertainty quantification
- **Multivariate dependencies**: Complex relationships
- **No training required**: Zero-shot imputation

</div>

</div>

Note:
Traditional imputation methods often make strong assumptions about the data distribution
or require careful feature engineering. Foundation models, in contrast, can adapt to
the specific patterns in your data through in-context learning.

--

<!-- Vertical Slide 2: Deep Dive into Anomaly Detection -->
## Anomaly Detection
### Deep Dive

<img src="assets/images/anomaly_detection_explanation.png" alt="Anomaly Detection Process" style="width: 90%; max-width: 1200px; margin: 0.5rem auto; display: block;">

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1rem; font-size: 0.75em;">

<div style="border: 2px solid #E74C3C; padding: 1rem; border-radius: 8px; background-color: #FADBD8;">

#### Traditional Approaches
- **Statistical**: Z-score, IQR (Interquartile Range)
- **Distance-Based**: LOF (Local Outlier Factor), k-NN
- **Density-Based**: DBSCAN
- **Isolation-Based**: Isolation Forest
- **Ensemble**: Multiple detectors combined

</div>

<div style="border: 2px solid #27AE60; padding: 1rem; border-radius: 8px; background-color: #E8F8F5;">

#### Foundation Model Approach
- **Likelihood-based**: Low probability = anomaly
- **In-context learning**: Adapts to data distribution
- **Multivariate**: Considers feature interactions
- **Zero-shot**: No training required

</div>

</div>

Note:
Traditional anomaly detection methods often struggle with high-dimensional data
and complex feature interactions. Foundation models naturally handle these challenges
through their transformer-based architecture and meta-learning approach.


--

<!-- Vertical Slide: Real-World Applications -->
### Real-World Applications

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1rem;">

<div style="border: 2px solid #3498DB; padding: 1.5rem; border-radius: 8px; background-color: #EBF5FB;">

#### 🧩 Missing Data Imputation

**🏥 Healthcare**
- Missing lab results in patient records
- Impute based on demographics and available tests

**💰 Finance**
- Incomplete transaction data
- Fill gaps using temporal and customer patterns

**📡 IoT & Sensors**
- Sensor data gaps due to failures
- Impute using correlated sensor readings

**📊 Survey Data**
- Non-responses in questionnaires
- Impute based on respondent characteristics

</div>

<div style="border: 2px solid #E67E22; padding: 1.5rem; border-radius: 8px; background-color: #FEF5E7;">

#### 🔍 Anomaly Detection

**💳 Fraud Detection**
- Credit card transactions
- Insurance claims and identity verification

**🏭 Quality Control**
- Manufacturing defects
- Product inspection and process monitoring

**🖥️ System Monitoring**
- Network intrusion detection
- Server performance and application errors

**🏥 Healthcare**
- Disease outbreak detection
- Unusual patient vitals and medical imaging

</div>

</div>

Note:
These real-world examples demonstrate the versatility of foundation models across both
imputation and anomaly detection tasks. The same model architecture can be applied across
different domains without modification, adapting to the specific patterns in each dataset
through in-context learning.
