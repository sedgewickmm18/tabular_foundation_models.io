<!-- Main Slide: Code Examples -->
## Foundational models applied

### Using tabular foundation models in real-life

<div style="margin-top: 30px;"></div>

<div style="line-height: 5.0;">

**Topics Covered**:
- When to use tabular foundation models and when not <br> _Hint: It won't guess your bank account number_
- Chunking - what to do when we have too much data.
- Prediction reconciliation
- Ontology - when we know more about the semantics of the data

</div>

<div style="margin-top: 30px;"></div>

Note:
These code examples demonstrate how to use TabPFN and TabICL effectively
in real-world applications, including handling large datasets and
optimizing performance.

--

<!-- Vertical Slide: When tabular foundation models work well -->
## When tabular foundation models work well

<div style="margin-top: 30px;"></div>

<div class="columns">
<div class="column">

### Data Structure Requirements

<div style="margin-top: 20px; font-size: 0.9em;">

**✅ Works Well: Time Series Data**
- Canonical ordering by time
- Sequential relationships matter
- Natural row order
- Example: Bank statements

**❌ Doesn't Work Well: Master Data**
- No canonical ordering
- Arbitrary row arrangement
- Order-independent records
- Example: Address books, catalogs

</div>

</div>
<div class="column">

![Data Structure Comparison](assets/images/data_structure_comparison.png)

</div>
</div>
<div style="text-align: center; margin: 40px auto 20px auto; max-width: 85%; font-size: 0.85em; padding: 20px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #6c757d;">

**Important Note on Ordering:**

Tabular foundation models do not rely on strict ordering and even need RoPE (rotary position encode) like features to express order by time, but they rely on relations between rows and features. In fact, apart from Mamba/Hydra based foundation models (a la Granite) the actual order does not matter

</div>


Note:
Foundational models like TabPFN work best with data that has a natural, canonical ordering (like time series where rows are ordered by time). They struggle with unordered data like address books or master data tables where row order is arbitrary and doesn't convey meaningful information. The diagram shows a bank statement with sequential trends (works well) versus an address book with no inherent order (doesn't work well).

--

<!-- Vertical Slide: On Tabular and Time Series Data -->
## On Tabular and Time Series Data

<div style="text-align: center;">
<img src="assets/images/fft_duality_diagram.png" alt="FFT Duality Diagram" style="max-width: 70%; height: auto;">
</div>

<div style="margin-top: 1.5rem; padding: 1rem; background-color: #f8f9fa; border: 2px solid #3498db; border-radius: 8px; font-size: 0.8em;">

Since tabular and time series data can be regarded as two sides of the same coin, the architecture foundation models for tabular and time series data also share common elements.

**Caveat**: Underlying assumption here is that the tabular data is canonically ordered.

</div>

Note:
This diagram illustrates the fundamental duality between tabular data and time series through FFT/IFFT transformations. Each row in a table can be viewed as frequency weights (volume knobs) that synthesize a time series signal via IFFT. Conversely, a time series window can be analyzed via FFT to extract frequency components that form a tabular row. This mathematical equivalence explains why transformer architectures work well for both domains.

--

<!-- Vertical Slide: Chunking Strategies -->
## Chunking strategies - when passing all data is computationally infeasible

<div style="display: flex; gap: 20px; align-items: flex-start; margin-top: 20px;">

<div style="flex: 1;">

<div style="border: 2px solid #2D6A4F; padding: 15px; border-radius: 8px; background-color: #F0FFF4; margin-bottom: 15px; font-size: 0.80em; text-align: left;margin-top: 5%;">

**Tumbling Windows**

**Concept**: Non-overlapping chunks
- Process each chunk independently
- Simple and fast

**Key Issue**: Boundary problems
- Context ends/starts abruptly at boundaries
- Missing values near edges lack full context
- Information loss at chunk transitions

</div>

<img src="assets/images/tumbling_window_chunking.svg" alt="Tumbling Window Chunking" style="width: 100%; margin: 0 auto; display: block;">

</div>

<div style="flex: 1;">

<div style="border: 2px solid #FF8C00; padding: 15px; border-radius: 8px; background-color: #FFF8E7; margin-bottom: 15px; font-size: 0.80em; text-align: left;">

**Sliding Windows**

**Concept**: Overlapping windows
- Each value seen in 3 contexts (start, middle, end)
- Complete coverage guaranteed

**Benefits**:
- No boundary information loss
- Multiple predictions per missing value
- Higher quality imputations

**Challenge**: Reconciliation needed!

</div>

<img src="assets/images/sliding_window_chunking.svg" alt="Sliding Window Chunking" style="width: 100%; margin: 0 auto; display: block;">

</div>

</div>

Note:
The sliding window approach (left) generates multiple predictions for each missing
value, requiring reconciliation. The tumbling window approach (right) is simpler
but loses context at boundaries. Sliding windows with reconciliation provide
higher quality results.

--

<!-- Vertical Slide: Log-Opinion Pool -->
## Reconciliation: Log-Opinion Pool
### Product of Experts

<div style="text-align: center; margin-bottom: 1em;">
<div style="display: inline-block; margin-right: 2em;">
<img src="assets/images/math/log_opinion_pool.svg" alt="Log Opinion Pool formula" style="max-width: 50%;">
</div>
<div style="display: inline-block;">
<strong>Equivalent to:</strong><br>
<img src="assets/images/math/log_opinion_pool_product.svg" alt="Log Opinion Pool product form" style="max-width: 50%;">
</div>
</div>

<div style="display: flex; gap: 20px; align-items: flex-start;">

<div style="flex: 1.2; border: 2px solid #6A4C93; padding: 15px; border-radius: 8px; background-color: #F5F0FF;">
<img src="assets/images/log_opinion_pooling_plot.png" alt="Log-Opinion Pool visualization" style="display: block; margin: 0 auto; max-width: 85%;">
</div>

<div style="flex: 0.8; border: 2px solid #2D6A4F; padding: 15px; border-radius: 8px; background-color: #F0FFF4; font-size: 0.85em;">

**Characteristics**:
- **Product** of probability distributions (in log space)
- Acts as **intersection of beliefs**
- More **conservative**: requires agreement across windows
- Preferred for foundation models

**Why Preferred?**
- If one window is **certain** (sharp peak), result stays sharp
- If one window is **uncertain** (flat), it doesn't dominate
- Better captures the consensus of expert predictions

</div>

</div>

Note:
The Log-Opinion Pool is mathematically preferred because it treats each
window as an expert that must agree. If any window is highly confident
about a particular value, that confidence is preserved in the final result.


--

<!-- Main Slide: Ontology -->
## Tabular Foundation Models and Ontology
### Leveraging Domain Knowledge

<div style="display: flex; gap: 20px; align-items: center;">

<!-- Left column: Existing content -->
<div style="flex: 1; font-size: 0.9em;">

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

</div>

<!-- Right column: Ontology example -->
<div style="flex: 1; font-size: 0.7em; text-align: left; border: 2px solid #2D6A4F; padding: 15px; border-radius: 8px; background-color: #F0FFF4;">

### Ontology example real estate - Leasing & Property Classifications

**Gewerbemietverträge** (Commercial lease contracts)
→ `contract_type`, `lease_category`

**Pauschalmiete** (Flat rent)
→ `rent_type`, `flat_rent_amount`

**Betriebskosten** (Operating costs)
→ `operating_costs`, `ancillary_costs`

**Immobilienklassen** (Property classes)
→ `property_class`, `asset_category`

**Mieterausbau** (Tenant improvements)
→ `tenant_improvements`, `fit_out_costs`

**Indexierung** (Indexation)
→ `index_type`, `adjustment_rate`

**Kündigungsfrist** (Notice period)
→ `notice_period_months`, `termination_date`

</div>

</div>

Note:
When you have a well-defined ontology for your data—understanding what each
feature represents and how they relate—you can significantly improve TabPFN's
performance through intelligent preprocessing and feature engineering. The real
estate example shows how German business terminology maps to structured data
columns, demonstrating the importance of domain knowledge in feature design.

--

<!-- Vertical Slide: Feature Engineering Based on Ontology -->
## Feature Engineering and Selection Based on Ontology

<div style="display: flex; gap: 20px; align-items: flex-start;">

<!-- Left text box -->
<div style="flex: 0 0 30%; font-size: 0.7em; text-align: left; border: 2px solid #2D6A4F; padding: 15px; border-radius: 8px; background-color: #F0FFF4;">

**TabPFN works best with smaller, highly informative feature sets**:
- ✅ Select features that **define the entity**
- ✅ Use domain knowledge to identify **core attributes**
- ✅ Remove redundant or low-signal features
- ❌ Avoid vast, high-dimensional raw data dumps

**Why This Works**:
- Foundation models excel at finding patterns in **meaningful** features
- Smaller feature sets → better attention → higher quality predictions
- Domain knowledge guides feature selection better than automated methods

</div>

<!-- Middle diagram (moved down) -->
<div style="flex: 0 0 35%; text-align: center; margin-top: 40px;">

<img src="assets/images/ontology_feature_workflow.png" alt="Ontology-Based Feature Engineering Workflow" style="width: 100%; border: 1px solid #ccc; border-radius: 8px;">

</div>

<!-- Right diagram (moved down slightly) -->
<div style="flex: 0 0 35%; text-align: center; margin-top: 20px;">

<img src="assets/images/famd_plot.png" alt="Factor Analysis of Mixed Data" style="width: 100%; border: 1px solid #ccc; border-radius: 8px;">

</div>

</div>

<div style="margin-top: 15px; font-size: 0.6em; color: #666; text-align: center; font-style: italic;">

Use knowledge about data (ontology) *and* feature analysis (PCA, MCA, FAMD) to reduce the number of features → less columns improve performance and accuracy of foundation models significantly

</div>

<div style="margin-top: 20px; font-size: 0.65em; color: #555;">

**Key Principle**: Use your ontology to guide preprocessing, and let TabPFN discover patterns within that structured representation.

</div>

Note:
The key insight is that TabPFN's in-context learning works best when each
feature carries significant information. Use your ontology to identify which
features truly define your entities and their relationships. The workflow diagram
shows how ontology-based filtering and semantic merging prepare data for FAMD
analysis, which then reveals the underlying structure in your feature space.

--

<!-- Vertical Slide: Ontology-Aware Preprocessing Strategies -->
## Ontology-Aware Preprocessing Strategies

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 0.75em;">

<div>

### Semantic Data Representation

**Problem**: Arbitrary encodings lose meaning
```python
# Poor
treatment = {'placebo': 1, 'low': 2, 'high': 3}

# Better: Preserve relationships
treatment = {'placebo': 0.0, 'low': 0.5, 'high': 1.0}
```

**Why it matters**:
- TabPFN infers relationships from input data
- Meaningful encodings → better pattern recognition
- Preserves domain structure

</div>

<div>

### Handle High Cardinality

**Challenge**: TabPFN has limitations with high-dimensional data

**Strategies**:
1. **Ontology-Based Grouping**: Roll up rare categories to parent
2. **Frequency Consolidation**: Merge rare categories into "Other"
3. **Hierarchical Encoding**: Multiple features at different levels

```python
# Example: ICD-10 codes (70K → 500 categories)
def group_codes(code, ontology):
    if ontology.get_frequency(code) < 0.01:
        return ontology.get_parent_category(code)
    return code
```

</div>

</div>

<div style="margin-top: 20px; font-size: 0.7em;">

### Complete Example: Clinical Trial Preprocessing

```python
class ClinicalTrialPreprocessor:
    def preprocess(self, df):
        # 1. Feature Selection: Use ontology to select core features
        core_features = self.ontology.get_entity_defining_features(entity_type='patient')
        df_selected = df[core_features]
        
        # 2. Semantic Encoding: Preserve treatment relationships
        df_selected['treatment_intensity'] = df_selected['treatment'].map(
            {'placebo': 0.0, 'low_dose': 0.5, 'high_dose': 1.0}
        )
        
        # 3. Handle High Cardinality: Group rare diagnoses
        df_selected['diagnosis_group'] = df_selected['diagnosis_code'].apply(
            lambda x: self.ontology.get_parent_category(x)
            if self.ontology.get_frequency(x) < 0.01 else x
        )
        return df_selected
```

</div>

Note:
These three strategies—semantic encoding, cardinality reduction, and feature selection—work together to create a preprocessing pipeline that preserves domain knowledge while making data suitable for TabPFN's in-context learning.
