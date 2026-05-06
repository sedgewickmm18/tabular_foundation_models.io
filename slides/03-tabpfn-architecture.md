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

<img src="assets/images/tabpfn_architecture.svg" alt="TabPFN Architecture Components" style="max-width: 70%; height: auto;">

<p style="font-size: 0.7em; margin-top: 10px;">
Similar to <a href="https://poloclub.github.io/transformer-explainer/" target="_blank">GPT-2's multi-layered transformer architecture</a>, TabPFN uses stacked transformer blocks for processing tabular data.
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

<!-- Vertical Slide: Mathematical Foundation -->
## Mathematical Formulation

### Bayesian Perspective
$$P(y|x, D_{train}) = \int P(y|x, \theta) P(\theta|D_{train}) d\theta$$

### Prior-Fitted Approach
$$P(\theta|D_{train}) \approx q_\phi(D_{train})$$

where $\phi$ is learned through meta-training

### In-Context Learning
Model learns to predict by observing training examples in context

Note:
The mathematical foundation is based on Bayesian inference, where the model
learns a distribution over model parameters given training data.

--

<!-- Vertical Slide: Training Process -->
## Training on Synthetic Data - Structural Causal Modeling

<div class="columns">
<div class="column">

### The SCM Process

**Step 1: Sampling Noise**
Every feature starts as a random draw from a distribution (e.g., Gaussian). This is the Exogenous component.

**Step 2: Building the DAG**
A random Directed Acyclic Graph is generated. Some features are "parents" and some are "children."

**Step 3: Sampling Functions**
For every node, a random mathematical function (MLPs, polynomials, or trig functions) is sampled to define how the children relate to the parents.

**Step 4: Creating the Dataset**
By running thousands of these unique SCMs, TabPFN creates a "Prior" of millions of synthetic tables. The model learns to predict $Y$ given $X$ across all these possible worlds.

</div>
<div class="column">

![SCM Example](assets/images/scm_diagram.svg)

</div>
</div>

Note:
Structural Causal Models (SCMs) provide a principled way to generate diverse synthetic datasets. Each SCM represents a different "world" with its own causal structure and functional relationships, enabling the model to learn general patterns that transfer to real data.