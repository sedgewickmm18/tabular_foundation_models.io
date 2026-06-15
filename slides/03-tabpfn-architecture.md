<!-- Main Slide: TabPFN Architecture -->
## TabPFN Architecture
### Prior-Fitted Networks for Tabular Data

**Core Concept**: Meta-learning on synthetic datasets

**Key Components**:
- Transformer encoder architecture
- Bayesian prior over datasets
- In-context learning mechanism
- Fast inference without training

Note:
TabPFN represents a breakthrough in tabular machine learning by applying
the transformer architecture with a novel training approach based on
Bayesian priors over datasets.


--

<!-- Vertical Slide: Architecture Details -->
## Architecture Components

<div class="columns" style="align-items: flex-start;">
<div class="column" style="flex: 0 0 65%; max-width: 65%;">

<img src="assets/images/tabpfn_architecture.svg" alt="TabPFN Architecture Components" style="max-width: 100%; height: auto;">

</div>
<div class="column" style="flex: 0 0 32%; max-width: 32%; display: flex; flex-direction: column; justify-content: center;">

<div class="model-code-box clickable-code-box" style="cursor: pointer; margin-top: 6em;">
<pre><code>TabImputeModel(
  # NxM --> Nx(M+12)x13x1024
  (feature_encoder): FeatureEncoder(
    (observed_linear_layer): Sequential(
      (0): Linear(in_features=1, out_features=1024, bias=True)
      (1): GELU(approximate='none')
      (2): Linear(in_features=1024, out_features=1024, bias=True)
    )
    (row_embedding): SinusoidalRowEmbedding()
    (column_embedding): SinusoidalColumnEmbedding()
    (row_cls_embedding): Embedding(12, 1024)
    (col_cls_embedding): Embedding(12, 1024)
  )
  (transformer_blocks): ModuleList(
    (0-11): 12 x TransformerEncoderLayer(
      (self_attention_between_datapoints): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
      )
      (self_attention_between_features): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
      )
      (linear1): Linear(in_features=1024, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=1024, bias=True)
      (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
      (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
      (norm3): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
      (gelu): GELU(approximate='none')
    )
  )
  (decoder): Decoder(
    (layers): ModuleList(
      (0): Linear(in_features=1024, out_features=1024, bias=True)
      (1): GELU(approximate='none')
      (2): Linear(in_features=1024, out_features=1024, bias=True)
      (3): GELU(approximate='none')
      (4): Linear(in_features=1024, out_features=5000, bias=True)
    )
  )
  # Nx(M+12)x13x1024 --> Nx(M+12)x13x5000 - last dimension for distribution
  # Impute: medians = self.bar_distribution.median(logits=preds)
)
</code></pre>
</div>

</div>
</div>

<p style="font-size: 0.7em; margin-top: 10px;">
Similar to <a href="https://poloclub.github.io/transformer-explainer/" target="_blank">GPT-2's multi-layered transformer architecture</a>, TabPFN uses stacked transformer blocks for processing tabular data.
It encodes tabular data into a continuous representation as shown above, applies multi-head self-attention to capture relationships between features, and outputs predictions through a final layer. The cells to be predicted are replaced with a dummy [BLANK] vector - same interpretation as for LLMs.
</p>

Note:
The TabPFN architecture consists of three main stages: Input Encoding handles feature normalization, positional encoding, and missing values; the Transformer Block (repeated N times) processes data through multi-head self-attention, feed-forward networks, and layer normalization; and the Output Layer produces classification heads, probability predictions, and uncertainty estimates. This architecture is carefully designed to handle the unique challenges of tabular data, including mixed feature types and missing values. Like GPT-2 and other transformer-based models, TabPFN leverages the power of multi-layered attention mechanisms, but specifically adapted for tabular data rather than text.

--

<!-- Vertical Slide: Detailed Transformer Architecture -->
## TabPFN Transformer Architecture
### Attention Mechanisms and Layer Structure

<div style="display: flex; flex-direction: column; gap: 5px; font-size: 0.65em;">
    <div>
        <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F51f38537-b8c6-4191-966f-1f65707ae208_2296x456.png" alt="TabPFN Architecture" style="max-width: 75%; height: auto;">
        <p style="font-size: 0.9em; margin: 2px 0; font-style: italic;">Source: <a href="https://mindfulmodeler.substack.com/p/the-architecture-behind-tabpfn">Christian Molnar's blog</a></p>
    </div>
    <div>
        <img src="assets/images/TabPFN-Attention.png" alt="TabPFN Attention Mechanism" style="max-width: 60%; height: auto;">
        <p style="font-size: 0.9em; margin: 2px 0; font-style: italic;">Attention across rows, attention across columns - the 2 dimensional aspect is a major difference to LLM (<a href="https://doi.org/10.1177/15330338251362050" title="MRI Delta-Radiomics and Morphological Feature-Driven TabPFN Model for Preoperative Prediction of Lymphovascular Invasion in Invasive Breast Cancer">DOI: 10.1177/15330338251362050</a>)</p>
    </div>
</div>

<div style="font-size: 0.6em; margin-top: 8px;">
<p style="margin: 3px 0;"><strong>Key Components:</strong></p>
<ul style="margin: 3px 0; padding-left: 20px;">
<li><strong>Feature attention</strong>: Attends to cells in the same row</li>
<li><strong>Datapoint attention</strong>: Attends to cells in the same column</li>
<li><strong>Multi-layer perceptron (MLP)</strong>: 2-layer fully-connected network</li>
<li><strong>Layer norm</strong>: Normalizes cell representations after each operation</li>
<li><strong>Skip connections</strong>: Residual layers before attention and MLP blocks</li>
</ul>
</div>

Note:
The TabPFN architecture uses a unique dual-attention mechanism. Feature attention
allows the model to understand relationships between different features for the same
datapoint (row-wise), while datapoint attention enables learning from similar examples
in the training set (column-wise). This design is specifically optimized for tabular
data where both feature interactions and example-based learning are crucial.

--

<!-- Vertical Slide: Training Process -->
## Training on Synthetic Data - Structural Causal Modeling

<div class="columns">
<div class="column" style="font-size: 0.85em; text-align: left;">

### The SCM Process

**Step 1: Sampling Noise**
Every feature starts as a random draw from a distribution (e.g., Gaussian). This is the Exogenous component.

<div style="height: 0.5em;"></div>

**Step 2: Building the DAG**
A random Directed Acyclic Graph is generated. Some features are "parents" and some are "children."

<div style="height: 0.5em;"></div>

**Step 3: Sampling Functions**
For every node, a random mathematical function (MLPs, polynomials, or trig functions) is sampled to define how the children relate to the parents.

<div style="height: 0.5em;"></div>

**Step 4: Creating the Dataset**
By running thousands of these unique SCMs, TabPFN creates a "Prior" of millions of synthetic tables. The model learns to predict Y given X across all these possible worlds.

</div>
<div class="column">

![SCM Example](assets/images/scm_diagram.svg)

</div>
</div>

Note:
Structural Causal Models (SCMs) provide a principled way to generate diverse synthetic datasets. Each SCM represents a different "world" with its own causal structure and functional relationships, enabling the model to learn general patterns that transfer to real data.

--

<!-- Vertical Slide: Loss Function -->
## TabPFN Loss Function

<div style="text-align: center; margin-bottom: 20px;">
<img src="assets/images/loss_function_decomposition.png" alt="Loss Function Decomposition" style="max-width: 64%; height: auto;">
<p style="font-size: 0.75em; margin-top: 5px; font-style: italic;">Source: <a href="https://mindfulmodeler.substack.com/p/how-tabular-foundation-models-are" target="_blank">Christian Molnar's blog - How Tabular Foundation Models Are Trained</a></p>
</div>

<div style="border: 2px solid #2D6A4F; padding: 15px; margin-bottom: 15px; text-align: left; font-size: 0.7em; line-height: 1.3; background-color: #F0FFF4;">

The term in brackets is the **plain NLL** (negative log likelihood) **between y_test and y_pred**.

The negative log-likelihood measures how well the predicted probability distribution matches the actual observed data. Lower NLL means better predictions.

<div style="height: 1.5em;"></div>

**NLL is equivalent to cross entropy** as measure of 'confusion' because y_test belongs to only one class where P(y_test) is 1 and else it's 0.

For classification tasks, the true label is a one-hot encoded vector (1 for the correct class, 0 for all others). This makes NLL mathematically equivalent to cross-entropy loss, which measures the "confusion" or divergence between predicted and true distributions.

<div style="height: 1.5em;"></div>

We can **get rid of the pesky integral** if we assume the generated training data from a particular task is a fixed constant for all tasks and approximate the integral with a sum over all tasks. The assumption is valid because the **SCM process is equivalent to Monte-Carlo sampling**.

Instead of computing an intractable integral over all possible tasks, we can approximate it by summing over a finite set of synthetic tasks generated by the Structural Causal Model (SCM) process. This Monte-Carlo approximation is valid because each synthetic dataset represents a sample from the task distribution.

</div>

<div style="border: 2px solid #1E40AF; padding: 15px; margin-bottom: 15px; text-align: left; font-size: 0.7em; line-height: 1.3; background-color: #EFF6FF;">

**Marginalizing over all tasks yields the posterior predictive distribution (PPD)**
**to predict target distribution from context and training data**

<div style="text-align: center; margin-top: 15px;">
<img src="assets/images/math/posterior_predictive_marginalized.svg" alt="Posterior Predictive Distribution Marginalized" style="max-width: 40%; height: auto;">
</div>

</div>

Note:
The loss function is the heart of TabPFN's training process. By minimizing the negative log-likelihood across millions of synthetic tasks, the model learns to perform approximate Bayesian inference. The key insight is that we can replace the intractable integral over all possible tasks with a practical sum over synthetic tasks generated by the SCM process, effectively teaching the transformer to marginalize over task uncertainty.


--

<!-- Vertical Slide: Binning for Regression -->
## Binning - how a classifier can do regression

<div class="columns">
<div class="column">

<div class="plot-container small">
  <!-- Interactive Plotly version (commented out due to annotation rendering issues in iframes) -->
  <!-- <iframe src="assets/interactive/quantile_binning.html"
          title="Quantile Binning Visualization">
    <img src="assets/images/quantile_binning.png" alt="Quantile Binning">
  </iframe> -->
  
  <!-- Static PNG version (reliable rendering) -->
  <img src="assets/images/quantile_binning.png"
       alt="Quantile Binning Visualization"
       style="width: 100%; height: auto;">
</div>

</div>
<div class="column" style="font-size: 0.75em; border: 2px solid #ccc; padding: 15px; text-align: left;">

### How it Enables Regression for a Classifier

**Binning (The Discrete Output)**: The continuous range of the target variable *y* is split into *B* discrete bins (typically 100). The classifier treats these bins as "classes."

**Probability Mass**: For a given input, the transformer outputs a probability distribution over these classes. Each class corresponds to a specific interval [y<sub>i</sub>, y<sub>i+1</sub>].

**Step Function PDF**: As shown in the diagram, these discrete probabilities form a piecewise constant approximation (a step function) of the true probability density function.

**Expected Value**: To get a final scalar prediction for regression, the model calculates the expected value of this distribution:

<img src="assets/images/math/expected_value_binning.svg" alt="Expected value formula" style="display: block; margin: 0.5em auto; max-width: 40%;">

**Uncertainty**: Because the model predicts a full distribution rather than just a point, it naturally provides uncertainty estimates (the "width" or variance of the steps).

</div>
</div>

<div style="font-size: 0.7em; margin-top: 20px; border: 2px solid #ccc; padding: 15px; text-align: left;">

### Quantile binning: From TabPFN to TabPFN-v2 

**Resolution**: By making bins smaller where data is dense, the model achieves higher precision in the most likely ranges of *y*.

**Outlier Robustness**: Wide bins at the edges (long tails) prevent a few extreme outliers from forcing the model to create hundreds of empty bins, which would happen with fixed-width binning.

**Uniform Loss**: It ensures that during training, the cross-entropy loss is more balanced because every "class" (bin) is represented by a similar number of samples in the prior.

</div>

Note:
TabPFN-v2 uses quantile binning to convert regression into a classification problem. This approach provides several advantages: it naturally handles uncertainty quantification, is robust to outliers, and ensures balanced training. The key insight is that by predicting a distribution over bins rather than a single value, the model can express uncertainty and achieve better calibration.
Structural Causal Models (SCMs) provide a principled way to generate diverse synthetic datasets. Each SCM represents a different "world" with its own causal structure and functional relationships, enabling the model to learn general patterns that transfer to real data.
