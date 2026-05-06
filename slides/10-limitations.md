<!-- Main Slide: Limitations -->
## Current Limitations & Future Work
### Understanding the Constraints

**Key Limitations**:
- Dataset size constraints (~1000 samples)
- Feature type limitations
- Computational requirements
- Interpretability challenges

**Future Directions**:
- Scalability improvements
- Multi-modal learning
- Enhanced interpretability
- Domain-specific models

Note:
Understanding limitations is crucial for appropriate application of
foundation models. Active research is addressing many of these constraints.

--

<!-- Vertical Slide: Detailed Limitations -->
## Detailed Limitations

### 1. Dataset Size Constraints
**Current**: ~1000 training samples (TabPFN), ~10K (TabICL)
**Impact**: Requires chunking for larger datasets
**Workaround**: Sliding window approach with reconciliation
**Future**: Improved architectures for larger contexts

### 2. Feature Types
**Current**: Primarily numerical and categorical
**Limitations**: 
- Text features require preprocessing
- Time series need special handling
- Images not directly supported
**Workaround**: Feature engineering and embeddings

### 3. Computational Requirements
**Current**: GPU recommended for best performance
**Impact**: Deployment constraints in resource-limited environments
**Workaround**: CPU mode available (slower)
**Future**: Model compression and optimization

### 4. Interpretability
**Current**: Black-box model, limited explainability
**Impact**: Difficult to understand predictions
**Workaround**: Post-hoc explanation methods (SHAP, LIME)
**Future**: Built-in interpretability features

Note:
Each limitation has practical workarounds, but they add complexity
to the deployment pipeline. Future research aims to address these
constraints directly in the model architecture.

--

<!-- Vertical Slide: Research Directions -->
## Active Research Directions

### Scalability
- **Efficient attention mechanisms**: Reduce memory footprint
- **Hierarchical processing**: Handle larger datasets
- **Distributed inference**: Parallel processing

### Multi-modal Learning
- **Text integration**: Natural language features
- **Time series**: Temporal patterns
- **Images**: Visual features
- **Graphs**: Relational data

### Interpretability
- **Attention visualization**: Understand feature importance
- **Counterfactual explanations**: What-if analysis
- **Rule extraction**: Interpretable patterns
- **Uncertainty quantification**: Confidence estimates

### Domain Adaptation
- **Few-shot fine-tuning**: Quick adaptation
- **Transfer learning**: Cross-domain knowledge
- **Meta-learning improvements**: Better generalization
- **Domain-specific priors**: Specialized models

Note:
The research community is actively working on these directions,
with new papers and improvements appearing regularly.

--

<!-- Vertical Slide: Community & Roadmap -->
## Community Contributions & Roadmap

### Open Source Ecosystem
- **GitHub**: Active development and issues
- **Papers**: Regular publications and preprints
- **Benchmarks**: Community-driven evaluations
- **Extensions**: Third-party tools and integrations

### Short-term Roadmap (6-12 months)
- ✅ Improved scalability (TabICL)
- 🔄 Better handling of categorical features
- 🔄 Enhanced uncertainty quantification
- 📋 Multi-task learning support

### Medium-term Roadmap (1-2 years)
- 📋 Multi-modal support (text, time series)
- 📋 Built-in interpretability features
- 📋 Federated learning capabilities
- 📋 AutoML integration

### Long-term Vision (2+ years)
- 📋 Universal tabular foundation model
- 📋 Real-time learning and adaptation
- 📋 Cross-domain transfer learning
- 📋 Automated feature engineering

Note:
The roadmap is ambitious but achievable given the rapid progress
in foundation models. Community contributions are welcome and
encouraged to accelerate development.

--

<!-- Vertical Slide: Best Practices Summary -->
## Best Practices & Recommendations

### When to Use Foundation Models
✅ **Ideal Scenarios**:
- Small to medium datasets (< 10K samples)
- Need for fast deployment
- Limited ML expertise
- Minimal tuning time available
- Mixed feature types
- Missing values present

❌ **Consider Alternatives**:
- Very large datasets (> 100K samples)
- Abundant computational resources
- Need for maximum accuracy (ensemble traditional methods)
- Strict interpretability requirements
- Real-time streaming data

### Deployment Checklist
1. ✅ Validate dataset size constraints
2. ✅ Check feature types compatibility
3. ✅ Test inference speed requirements
4. ✅ Evaluate baseline performance
5. ✅ Consider chunking strategy if needed
6. ✅ Plan for model updates and monitoring
7. ✅ Document limitations and assumptions

### Getting Started
1. Start with zero-shot predictions
2. Evaluate performance on validation set
3. Consider fine-tuning if needed
4. Implement chunking for large datasets
5. Monitor performance in production
6. Iterate based on feedback

Note:
Following these best practices will help ensure successful deployment
of foundation models in production environments.