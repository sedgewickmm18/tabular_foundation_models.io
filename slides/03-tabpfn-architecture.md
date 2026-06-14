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

<!-- Vertical Slide: Predictive Posterior Distribution -->
## Predictive posterior distribution

<div style="display: flex; gap: 20px; margin-bottom: 20px;">

<!-- Left box: KL Divergence Decomposition -->
<div style="flex: 1; border: 2px solid #2D6A4F; padding: 15px; text-align: center; font-size: 0.65em; line-height: 1.2; background-color: #F0FFF4;">

<h3 style="margin-top: 0; font-size: 1.1em; color: #2D6A4F;">How NLL, KL Divergence, Cross Entropy and Entropy are connected*</h3>

<img src="assets/images/kl_nll_entropy_diagram.png" alt="KL Divergence Decomposition" style="max-width: 100%; height: auto;">

<p style="margin-top: 10px; text-align: left;">
<strong>Key Insight:</strong> Minimizing NLL (cross-entropy) is equivalent to minimizing reverse KL divergence D<sub>KL</sub>(p || q) because the entropy H(p) is constant.
</p>

<p style="margin-top: 10px; text-align: left; font-size: 0.9em; font-style: italic;">
*LogSoftmax → NLL Loss ~ raw logits → Cross Entropy
</p>

</div>

<!-- Right box: Reverse KL Properties -->
<div style="flex: 1; border: 2px solid #1E40AF; padding: 15px; text-align: center; font-size: 0.65em; line-height: 1.2; background-color: #EFF6FF;">

<h3 style="margin-top: 0; font-size: 1.1em; color: #1E40AF;">Reverse KL Preserves Information</h3>

<img src="assets/images/reverse_forward_kl_comparison.png" alt="Reverse vs Forward KL" style="max-width: 100%; height: auto;">

<p style="margin-top: 10px; text-align: left;">
<strong>Why Reverse KL?</strong> With true distribution p as the second argument, reverse KL forces Q to avoid regions where P = 0, preserving information. Forward KL would spread Q across all modes, losing precision.
</p>

</div>

</div>

<div style="border: 2px solid #4a90e2; padding: 15px; margin-top: 20px; text-align: left; font-size: 0.65em; line-height: 1.2; background-color: #f0f8ff;">

In Prior-Data Fitted Networks such as TabPFN, the "true" distribution <em>p</em> is the actual Bayesian posterior defined by the prior you've chosen. By minimizing NLL over millions of synthetic datasets sampled from that prior, the Transformer is forced to close the KL gap, effectively "learning" to perform Bayesian inference without ever explicitly calculating an integral or using Bayes' theorem during inference.

</div>

Note:
The KL divergence measures how one probability distribution differs from another. Since the entropy H(p) of the true data distribution is constant (independent of our model), minimizing the cross-entropy H(p,q) (which is the NLL) is equivalent to minimizing the KL divergence between the true and predicted distributions.

--

<!-- Vertical Slide: Posterior Predictive Visualization -->
## Posterior Predictive Distribution - a Bayesian RANSAC like example

<div style="border: 2px solid #ccc; padding: 8px; margin-top: 2px; margin-bottom: 10px; text-align: left; font-size: 0.55em; line-height: 1.1;">

<span>Now we look for a joint minimum across the entire task distribution 𝝉. The training objective is effectively:</span>

<div style="text-align: center; margin: 0.5em 0;">
<img src="assets/images/math/training_objective_tasks.svg" alt="Training Objective" style="max-width: 30%;">
</div>

<span>with 𝜽 the transformer weights, 𝝉 the set of tasks and 𝒟 data generated by a task.</span>

<div style="height: 0.5em;"></div>

<span>This is effectively marginalizing over all tasks to get the posterior predictive distribution.</span>

<div style="text-align: center; margin: 0.5em 0;">
<img src="assets/images/math/posterior_predictive_marginalized.svg" alt="Posterior Predictive Distribution" style="max-width: 30%;">
</div>

</div>

<div style="text-align: center;">

![Posterior Predictive Example](assets/images/posterior_predictive_example.png)

</div>

<div style="border: 2px solid #ccc; padding: 8px; margin-top: 10px; text-align: left; font-size: 0.65em; line-height: 1.2;">

**Left**: Training data 𝒟 with 8 points following a linear relationship.

**Middle**: 30 candidate tasks 𝒫(y|x,τ) with different task parameters τ sampled from the prior. Line opacity represents posterior probability p(τ|𝒟) - tasks that fit the data better are more opaque. This is analogous to Bayesian RANSAC, where we maintain a distribution over hypotheses.

**Right**: The posterior predictive distribution 𝒫(y|x_new,𝒟) at x_new=4.5, obtained by marginalizing over all tasks weighted by their posterior probabilities. This shows the uncertainty in the prediction.

</div>

Note:
This visualization demonstrates the core concept of Bayesian inference: instead of selecting a single "best" model, we maintain a distribution over all possible models weighted by how well they explain the observed data. The final prediction naturally incorporates uncertainty from both model uncertainty and data noise. TabPFN learns to approximate this marginalization process through its transformer architecture trained on millions of synthetic tasks.
---


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
