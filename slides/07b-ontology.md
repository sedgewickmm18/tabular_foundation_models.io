<!-- Main Slide: Ontology -->
## Tabular Foundation Models and Ontology
### Leveraging Domain Knowledge

**When Ontology is Known**:
- Domain structure is well-defined
- Feature relationships are understood
- Semantic meaning is clear
- Entity definitions are established

**Key Strategies**:
- Feature engineering based on domain knowledge
- Semantic data representation
- Handling high cardinality features
- Ontology-aware preprocessing

Note:
When you have a well-defined ontology for your data—understanding what each
feature represents and how they relate—you can significantly improve TabPFN's
performance through intelligent preprocessing and feature engineering.

--

<!-- Vertical Slide: Feature Engineering Based on Ontology -->
## Feature Engineering and Selection Based on Ontology

### Prioritize Informative Features

**TabPFN works best with smaller, highly informative feature sets**:
- ✅ Select features that **define the entity** (e.g., patient demographics in clinical trials)
- ✅ Use domain knowledge to identify **core attributes**
- ✅ Remove redundant or low-signal features
- ❌ Avoid vast, high-dimensional raw data dumps

**Example: Clinical Trial Data**
```python
# Instead of 500+ raw features
raw_features = ['age', 'gender', 'weight', 'height', 'bmi', 
                'blood_pressure_systolic', 'blood_pressure_diastolic',
                'lab_test_1', 'lab_test_2', ..., 'lab_test_200']

# Select core entity-defining features (20-50 features)
core_features = [
    'age', 'gender', 'bmi',  # Demographics
    'primary_diagnosis', 'comorbidity_count',  # Clinical status
    'baseline_biomarker_1', 'baseline_biomarker_2',  # Key biomarkers
    'treatment_arm', 'prior_treatment_history'  # Treatment context
]
```

**Why This Works**:
- Foundation models excel at finding patterns in **meaningful** features
- Smaller feature sets → better attention → higher quality predictions
- Domain knowledge guides feature selection better than automated methods

Note:
The key insight is that TabPFN's in-context learning works best when each
feature carries significant information. Use your ontology to identify which
features truly define your entities and their relationships.

--

<!-- Vertical Slide: Semantic Data Representation -->
## Semantic Data Representation

### Beyond Simple Encoding

**Problem with Raw Encoding**:
```python
# Poor: Arbitrary numeric encoding loses semantic meaning
treatment_type = {
    'placebo': 1,
    'low_dose': 2,
    'high_dose': 3
}
```

**Better: Semantic Representation**:
```python
# Option 1: Ordinal encoding that preserves relationships
treatment_intensity = {
    'placebo': 0.0,      # No treatment
    'low_dose': 0.5,     # Moderate treatment
    'high_dose': 1.0     # Maximum treatment
}

# Option 2: Multiple binary features capturing semantics
features = {
    'is_treated': [0, 1, 1],           # Treatment vs placebo
    'dose_level': [0, 0.5, 1.0],       # Intensity if treated
    'treatment_duration_weeks': [0, 12, 12]  # Duration
}

# Option 3: Embeddings from domain knowledge
# Use pre-trained medical concept embeddings
treatment_embedding = medical_ontology.get_embedding('high_dose_chemotherapy')
```

**Why Semantic Representation Matters**:
- TabPFN infers relationships from input data
- Meaningful encodings → better pattern recognition
- Preserves domain structure in the feature space

Note:
The way you encode categorical variables can dramatically affect TabPFN's
ability to learn relationships. Use your ontology to create encodings that
preserve semantic meaning and natural orderings.

--

<!-- Vertical Slide: Handling High Cardinality Features -->
## Handle High Cardinality Features

### Strategies for Many-Category Variables

**Challenge**: TabPFN 2.5 has limitations with extremely high-dimensional data

**Strategy 1: Ontology-Based Grouping**
```python
# Example: Medical diagnosis codes (ICD-10)
# Original: 70,000+ unique codes
# Grouped by ontology hierarchy

def group_diagnosis_codes(icd10_code, ontology):
    """Group rare diagnoses into broader categories"""
    if ontology.get_frequency(icd10_code) < 0.01:  # Rare diagnosis
        # Roll up to parent category
        return ontology.get_parent_category(icd10_code)
    return icd10_code

# Result: 500 meaningful categories instead of 70,000
```

**Strategy 2: Frequency-Based Consolidation**
```python
def consolidate_rare_categories(df, column, threshold=0.01):
    """Consolidate rare categories into 'Other'"""
    value_counts = df[column].value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold].index
    
    df[column] = df[column].apply(
        lambda x: 'Other' if x in rare_categories else x
    )
    return df
```

**Strategy 3: Hierarchical Encoding**
```python
# Use ontology hierarchy to create multiple features
# Example: Geographic location
location_features = {
    'country': 'USA',           # Top level
    'region': 'Northeast',      # Mid level
    'state': 'Massachusetts',   # Specific level
    'is_urban': True           # Derived attribute
}
# Instead of one feature with 50,000 cities
```

Note:
High cardinality features can overwhelm TabPFN's context window. Use your
ontology to intelligently reduce dimensionality while preserving the most
important semantic distinctions.

--

<!-- Vertical Slide: Practical Example -->
## Practical Example: Clinical Trial Data

### Complete Ontology-Aware Preprocessing

<div style="position: relative;">

<div style="font-size: 0.65em; max-width: 70%; margin: 0 auto;">

```python
import pandas as pd
from tabpfn import TabPFNClassifier

class ClinicalTrialPreprocessor:
    def __init__(self, ontology):
        self.ontology = ontology
    
    def preprocess(self, df):
        """Apply ontology-aware preprocessing"""
        # 1. Feature Selection
        core_features = self.ontology.get_entity_defining_features(
            entity_type='patient', importance_threshold=0.7
        )
        df_selected = df[core_features]
        
        # 2. Semantic Encoding
        df_selected['treatment_intensity'] = df_selected['treatment'].map({
            'placebo': 0.0, 'low_dose': 0.5, 'high_dose': 1.0
        })
        
        # 3. Handle High Cardinality
        df_selected['diagnosis_group'] = df_selected['diagnosis_code'].apply(
            lambda x: self.ontology.get_parent_category(x)
            if self.ontology.get_frequency(x) < 0.01 else x
        )
        
        return df_selected

# Usage
preprocessor = ClinicalTrialPreprocessor(medical_ontology)
X_processed = preprocessor.preprocess(X_raw)
model = TabPFNClassifier(device='cuda')
model.fit(X_processed, y)
```

</div>

<div style="position: absolute; top: 10%; right: 5%; width: 25%; text-align: left; font-size: 0.75em; color: #2D6A4F; background-color: #F0FFF4; padding: 15px; border-radius: 8px; border: 2px solid #2D6A4F; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
<div style="font-weight: bold; margin-bottom: 8px;">💡 LLM Opportunity</div>
<div style="border-left: 3px solid #2D6A4F; padding-left: 10px;">
Here LLM-based automated code generation comes into play...
</div>
<div style="margin-top: 10px; font-size: 0.9em; color: #555;">
<svg width="40" height="40" style="position: absolute; left: -45px; top: 50%; transform: translateY(-50%);">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#2D6A4F" />
    </marker>
  </defs>
  <path d="M 40 20 Q 20 20, 5 20" stroke="#2D6A4F" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
</svg>
</div>
</div>

</div>

Note:
This example shows how to combine all three strategies—feature selection,
semantic encoding, and cardinality reduction—into a cohesive preprocessing
pipeline guided by your domain ontology. The annotation highlights where
LLM-based code generation can assist in creating ontology-aware preprocessing.

--

<!-- Vertical Slide: Best Practices Summary -->
## Best Practices: Ontology-Driven Approach

### Guidelines for Success

**✅ DO**:
- **Use domain expertise** to select 20-50 most informative features
- **Preserve semantic relationships** in encodings (ordinal, hierarchical)
- **Group rare categories** using ontology hierarchy
- **Create derived features** that capture domain knowledge
- **Document your ontology** and preprocessing decisions
- **Validate** that semantic encodings improve performance

**❌ DON'T**:
- Include hundreds of raw features without selection
- Use arbitrary numeric encodings (1, 2, 3) for unordered categories
- Keep extremely rare categories that add noise
- Ignore domain structure when preprocessing
- Apply generic preprocessing without considering semantics

### Key Principle

> **"Let your ontology guide your preprocessing, and let TabPFN discover the patterns within that structured representation."**

Note:
The most successful applications of TabPFN combine the model's powerful
in-context learning with thoughtful, ontology-driven preprocessing. Your
domain knowledge is a crucial input to the system, not something to be
ignored in favor of raw data.

--

<!-- Vertical Slide: When to Apply Ontology-Aware Preprocessing -->
## When to Apply This Approach

### Decision Framework

**High Value Scenarios** 🎯:
- ✅ Well-established domain with clear ontology (medical, legal, financial)
- ✅ Features have known semantic relationships
- ✅ High cardinality categorical variables present
- ✅ Expert knowledge available for feature engineering
- ✅ Entity definitions are clear and stable

**Lower Value Scenarios** ⚠️:
- ❌ Exploratory analysis with unknown domain structure
- ❌ Features are already well-preprocessed
- ❌ Low cardinality, simple feature sets
- ❌ No domain expertise available
- ❌ Rapidly changing feature definitions

### Trade-offs

**Benefits**:
- Better model performance
- More interpretable features
- Reduced dimensionality
- Faster inference

**Costs**:
- Requires domain expertise
- Additional preprocessing complexity
- Maintenance of ontology mappings
- Risk of encoding biases

Note:
Ontology-aware preprocessing is most valuable when you have a mature
understanding of your domain and can invest in thoughtful feature engineering.
For exploratory work or simple datasets, the additional complexity may not
be justified.