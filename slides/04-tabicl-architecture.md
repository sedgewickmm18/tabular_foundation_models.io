<!-- Main Slide: TabICL Architecture -->
## TabICL: Enhanced In-Context Learning
### Successor to TabPFN

**Key Improvements**:
- Better scalability for larger datasets
- Enhanced in-context learning
- Improved performance
- More efficient inference

Note:
TabICL builds on TabPFN's foundation, addressing its limitations and
extending its capabilities to handle larger and more complex datasets.

--

<!-- Vertical Slide: Architectural Differences -->
## Architectural Enhancements

<div style="text-align: center;">
  <img src="assets/images/architecture_comparison.png" alt="TabPFN vs TabICL Architecture" style="max-width: 65%; max-height: 45vh;">
</div>

<div style="font-size: 0.65em;">

- **Stage 1 (Row Embedder)**: Acts as a specialized pre-processor that looks only at features of a single row and "summarizes" them into a single mathematical vector (a "token")
- **Stage 2 (Sequential ICL)**: Treats the dataset exactly like an LLM treats a sentence - instead of a sequence of words, it sees a sequence of row tokens
- **Impact**: The main Transformer in Stage 2 never sees the original features, only the summaries. This allows it to handle much larger datasets (more rows) because it is no longer distracted by high feature counts

For deeper coverage of TabICL improvements, such as numerical complexity of row-wise encodings and column based embeddings that capture statistical properties, see [Gael Varoquaux' blog entry](https://gael-varoquaux.info/science/tabicl-pretraining-the-best-tabular-learner.html)

</div>

Note:
The two-stage architecture separates feature processing from sequential learning, enabling better scalability and performance.

--

<!-- Vertical Slide: Mathematical Improvements -->
## Mathematical Enhancements

<div style="display: flex; gap: 20px; font-size: 0.74em; align-items: stretch;">

<div style="flex: 1; border: 2px solid #C1121F; padding: 15px; border-radius: 8px; background-color: #FFF5F5;">

**TabPFN: Binned Classification Approach**

TabPFN does not treat regression as a continuous value prediction. Instead, it turns regression into a classification problem:

- **Mechanism**: It discretizes the target variable into a fixed number of bins (buckets)
- **Loss**: It applies Cross-Entropy Loss over these bins
- **Benefit**: This allows the model to predict a full probability distribution (e.g., "The value is 60% likely to be in bin A and 40% in bin B"), which naturally handles uncertainty

</div>

<div style="flex: 1; border: 2px solid #2D6A4F; padding: 15px; border-radius: 8px; background-color: #F0FFF4;">

**TabICL v2: Quantile Regression**

TabICL v2 moved away from binning to use Quantile Regression:

- **Mechanism**: The model predicts a high number of individual quantiles (typically 999 levels)
- **Loss**: It uses Pinball Loss (also known as Quantile Loss) summed across all 999 levels
- **Benefit**: It avoids the "resolution" issues of fixed bins. By averaging these 999 quantiles at inference time, it produces a highly accurate and flexible point estimate while still providing a dense picture of the prediction's uncertainty

</div>

</div>

<div class="plot-container" style="margin-top: 18px;">
  <iframe src="assets/interactive/compare_loss_functions.html"
          title="Comparison of cross-entropy loss and pinball loss"
          style="width: 100%; height: 500px; border: none;">
    <!-- Fallback for browsers without iframe support -->
    <img src="assets/images/compare_loss_functions.png"
         alt="Comparison of cross-entropy loss and pinball loss"
         style="max-width: 100%; height: auto;">
  </iframe>
</div>

<div style="margin-top: 12px; font-size: 0.75em; text-align: center;">
Reference: <a href="https://arxiv.org/html/2602.11139v1" target="_blank">https://arxiv.org/html/2602.11139v1</a>
</div>

Note:
The shift from binned classification to quantile regression represents a fundamental improvement in how TabICL handles regression tasks, providing better accuracy and uncertainty quantification.

--

<!-- Vertical Slide: Comparison -->
## TabPFN vs TabICL

| Feature | TabPFN | TabICL |
|---------|--------|--------|
| **Max Dataset Size** | ~1000 samples | ~10000 samples |
| **Inference Speed** | Fast | Faster |
| **Memory Usage** | Moderate | Lower |
| **Accuracy** | High | Higher |
| **Ease of Use** | Simple | Simple |

Note:
TabICL maintains TabPFN's simplicity while significantly improving
performance and scalability.
