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