# Chapter 230: Membership Inference

## 1. Introduction

Membership inference is a class of privacy attacks against machine learning models. The central question is deceptively simple: **given a trained model and a specific data point, can we determine whether that data point was part of the model's training set?**

At first glance, a well-generalized model should treat seen and unseen data identically. In practice, however, models memorize aspects of their training data, and this memorization leaks through observable outputs such as prediction confidences, loss values, and label distributions. These subtle statistical signatures form the basis of membership inference attacks.

In the context of algorithmic trading, membership inference poses unique and severe risks. A trading model's training data often encodes proprietary information: which assets a fund monitors, what data sources it relies upon, and the historical period over which it was calibrated. An adversary who can infer membership can reverse-engineer strategic intelligence that funds spend millions to develop and protect.

This chapter provides a rigorous treatment of membership inference: its mathematical foundations, practical attack methods, defenses, and a complete Rust implementation that demonstrates every concept against real cryptocurrency market data from Bybit.

## 2. Mathematical Foundation

### 2.1 Loss-Based Thresholding

The simplest membership inference attack exploits the fact that models tend to have lower loss on training (member) data than on unseen (non-member) data. Given a model $f_\theta$, a data point $(x, y)$, and a loss function $\mathcal{L}$, the attack is:

$$
\text{member}(x, y) = \mathbb{1}[\mathcal{L}(f_\theta(x), y) < \tau]
$$

where $\tau$ is a threshold chosen to maximize attack accuracy. The threshold can be calibrated on a hold-out set or selected analytically. This approach works because stochastic gradient descent drives the loss on training examples below the loss on the general population.

The effectiveness of this attack is directly related to the generalization gap: the difference between training loss and test loss. Models that overfit heavily are far more vulnerable.

### 2.2 Shadow Model Approach (Shokri et al., 2017)

Shokri et al. introduced the shadow model framework, which remains the most influential membership inference methodology. The key insight is to train auxiliary models (shadow models) that mimic the behavior of the target model, then use these shadow models to generate labeled training data for a binary attack classifier.

The procedure is:

1. **Shadow model training**: Train $k$ shadow models $f_1, \ldots, f_k$ on datasets $D_1, \ldots, D_k$ drawn from a distribution similar to the target's training distribution.
2. **Attack dataset construction**: For each shadow model $f_i$ and each data point $(x, y)$:
   - If $(x, y) \in D_i$, label the output vector $f_i(x)$ as "member."
   - If $(x, y) \notin D_i$, label the output vector $f_i(x)$ as "non-member."
3. **Attack model training**: Train a binary classifier $\mathcal{A}$ on the attack dataset.
4. **Inference**: Given the target model's output $f_\theta(x)$ for a query point, apply $\mathcal{A}(f_\theta(x))$ to predict membership.

Formally, the attack model learns:

$$
\mathcal{A}: \mathbb{R}^c \times \mathcal{Y} \rightarrow \{0, 1\}
$$

where $c$ is the number of output classes and $\mathcal{Y}$ is the label space.

### 2.3 Likelihood Ratio Test

A more principled approach frames membership inference as a hypothesis test:

- $H_0$: $(x, y)$ was **not** in the training set.
- $H_1$: $(x, y)$ **was** in the training set.

The likelihood ratio is:

$$
\Lambda(x, y) = \frac{P(\text{output} \mid (x,y) \in D_{\text{train}})}{P(\text{output} \mid (x,y) \notin D_{\text{train}})}
$$

When $\Lambda(x, y) > 1$, the evidence favors membership. This approach, formalized by Carlini et al. (2022), achieves state-of-the-art results by estimating these likelihoods through reference models trained with and without the target point.

### 2.4 Label-Only Attacks

In many deployment scenarios, models return only hard labels (e.g., "buy" or "sell") rather than confidence scores. Label-only attacks exploit the decision boundary geometry:

1. **Perturbation-based**: Measure how much perturbation $\delta$ is needed to change the model's prediction. Training points tend to be further from the decision boundary, requiring larger perturbations.
2. **Transfer-based**: Train a substitute model that provides soft outputs, then apply standard confidence-based attacks to the substitute.

The perturbation distance for a point $x$ is:

$$
d(x) = \min_{\delta} \|\delta\|_2 \quad \text{s.t.} \quad f_\theta(x + \delta) \neq f_\theta(x)
$$

Members tend to have larger $d(x)$ because the model has "carved out" a confident region around them.

## 3. Privacy Risks in Trading

### 3.1 Inferring Trading Universe

If an adversary can determine which assets' data were used to train a model, they can infer the fund's **trading universe**. For a multi-asset model, membership inference on per-asset features reveals which instruments the fund actively monitors and likely trades.

### 3.2 Revealing Proprietary Data Sources

Quantitative funds invest heavily in alternative data: satellite imagery, credit card transactions, social sentiment, and more. If a model's behavior on alternative data features differs meaningfully from its behavior on public data, membership inference can reveal whether a fund uses a specific proprietary data source.

### 3.3 Identifying the Training Period

By querying a model with data from different time periods and observing membership signals, an adversary can estimate the model's training window. This reveals how frequently the fund retrains, whether it uses recent data, and potentially the fund's lookback horizon - all strategically valuable intelligence.

### 3.4 Front-Running and Strategy Reconstruction

In the most adversarial scenario, an attacker who knows which data points trained a model can reconstruct significant aspects of the trading strategy itself. If the attacker determines that a model was trained on order book imbalance features from specific venues during specific periods, they can approximate the strategy's signals and front-run its trades.

## 4. Attack Methods

### 4.1 Confidence-Based Attack

The confidence-based attack uses the maximum prediction probability as a membership signal:

```
member(x) = 1 if max(f_theta(x)) > tau
```

Training examples typically receive higher-confidence predictions because the model has been optimized to classify them correctly. The threshold $\tau$ is calibrated to balance precision and recall.

**Implementation detail**: For regression models common in trading (predicting returns), we transform the continuous output into a confidence measure using the prediction error relative to a baseline.

### 4.2 Loss-Based Attack

The loss-based attack directly uses the model's loss:

```
member(x, y) = 1 if L(f_theta(x), y) < tau
```

This is often the most effective single-metric attack because loss directly reflects the optimization objective. The model has been explicitly trained to minimize loss on member data.

### 4.3 Shadow Model Training

The shadow model approach requires:

1. A dataset from a similar distribution to the target's training data.
2. Sufficient compute to train multiple shadow models.
3. A separate attack classifier (often a neural network or logistic regression).

In trading contexts, shadow models can be trained on publicly available market data. Even if the target model uses proprietary features, shadow models trained on related public features can produce effective attack classifiers.

### 4.4 Metric-Based Attacks (Entropy and Modified Entropy)

**Entropy attack**: Uses the entropy of the prediction distribution as the signal:

$$
H(f_\theta(x)) = -\sum_i f_\theta(x)_i \log f_\theta(x)_i
$$

Member data points tend to produce lower-entropy (more concentrated) predictions.

**Modified entropy attack**: Incorporates the true label by weighting the entropy:

$$
H_{\text{mod}}(f_\theta(x), y) = -\left(1 - f_\theta(x)_y\right) \log f_\theta(x)_y - \sum_{i \neq y} f_\theta(x)_i \log(1 - f_\theta(x)_i)
$$

This captures both the model's confidence in the correct class and its uncertainty across incorrect classes.

## 5. Defenses

### 5.1 Regularization

L2 regularization (weight decay) adds a penalty $\lambda \|\theta\|_2^2$ to the loss function, discouraging the model from memorizing individual training examples. By constraining the model's capacity, regularization reduces the generalization gap and consequently the membership signal.

The regularized objective becomes:

$$
\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \lambda \|\theta\|_2^2
$$

Empirically, increasing $\lambda$ monotonically decreases membership inference accuracy, but at the cost of model utility.

### 5.2 Differential Privacy

Differential privacy (DP) provides a formal mathematical guarantee. A mechanism $\mathcal{M}$ is $(\epsilon, \delta)$-differentially private if for any neighboring datasets $D, D'$ (differing in one element) and any set of outputs $S$:

$$
P(\mathcal{M}(D) \in S) \leq e^{\epsilon} P(\mathcal{M}(D') \in S) + \delta
$$

DP-SGD adds calibrated Gaussian noise to clipped gradients during training, bounding the influence of any single training example. The privacy budget $\epsilon$ directly limits membership inference advantage.

### 5.3 Knowledge Distillation

Knowledge distillation transfers knowledge from a "teacher" model to a "student" model by training the student on the teacher's soft outputs rather than the original labels. This process naturally sanitizes the training data's influence:

1. Train the teacher on the sensitive dataset.
2. Generate soft labels from the teacher on a public transfer set.
3. Train the student on the transfer set with soft labels.

The student model never directly sees the sensitive training data, significantly reducing membership leakage.

### 5.4 Output Perturbation

Adding noise to the model's output predictions reduces the precision of the signals that membership inference attacks rely on:

$$
\tilde{f}_\theta(x) = f_\theta(x) + \mathcal{N}(0, \sigma^2 I)
$$

The noise scale $\sigma$ must be large enough to obscure membership signals but small enough to preserve prediction utility. Adaptive calibration based on the prediction's entropy can optimize this trade-off.

## 6. Implementation Walkthrough (Rust)

Our Rust implementation in `rust/src/lib.rs` provides a complete membership inference framework:

### Target Model

We implement a simple linear regression model that can be trained on member data and evaluated on both member and non-member data. The model uses gradient descent with configurable L2 regularization.

```rust
pub struct LinearModel {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub l2_lambda: f64,
}
```

### Attack Pipeline

The attack pipeline follows a systematic structure:

1. **Data splitting**: Divide available data into member and non-member sets.
2. **Model training**: Train the target model on member data only.
3. **Signal extraction**: Compute loss, confidence, and entropy metrics for all data points.
4. **Threshold calibration**: Find optimal thresholds that maximize attack accuracy.
5. **Evaluation**: Measure attack success with accuracy, precision, recall, and AUC.

### Shadow Model Approach

Our shadow model implementation trains multiple models on random subsets, collects prediction features, and trains a logistic regression attack classifier:

```rust
pub struct ShadowModelAttack {
    pub num_shadows: usize,
    pub attack_weights: Array1<f64>,
    pub attack_bias: f64,
}
```

### Defense Evaluation

The defense module demonstrates how increasing L2 regularization strength reduces attack accuracy while potentially degrading model utility.

## 7. Bybit Data Integration

The implementation includes a `BybitClient` that fetches real OHLCV data from the Bybit public API:

```rust
pub struct BybitClient {
    base_url: String,
}
```

The client fetches kline (candlestick) data for any trading pair and interval. For our experiments, we use BTCUSDT with 1-hour candles, extracting features such as:

- Log returns: $r_t = \log(p_t / p_{t-1})$
- Normalized volume: $v_t / \bar{v}$
- High-low range: $(h_t - l_t) / p_t$
- Open-close spread: $(c_t - o_t) / o_t$

These features are split into member and non-member sets by time period, simulating a scenario where an attacker tries to determine which historical period was used for training.

The trading example (`rust/examples/trading_example.rs`) demonstrates the full pipeline:

1. Fetch BTCUSDT data from Bybit.
2. Engineer features from raw OHLCV data.
3. Split into member (training) and non-member (holdout) sets.
4. Train a target model on member data.
5. Execute all four attack methods.
6. Apply L2 regularization defense and measure the reduction in attack success.

## 8. Key Takeaways

1. **Membership inference is a real threat**: Even simple models leak information about their training data through prediction confidence, loss, and entropy.

2. **Trading models are especially vulnerable**: The high-value, proprietary nature of trading data makes membership inference attacks particularly damaging in financial contexts.

3. **Multiple attack vectors exist**: From simple threshold-based attacks to sophisticated shadow model approaches, adversaries have a rich toolkit. Defending against one method does not guarantee safety against others.

4. **The generalization gap is the root cause**: Membership inference fundamentally exploits the difference between a model's behavior on training vs. unseen data. Any technique that reduces this gap (regularization, early stopping, data augmentation) helps.

5. **Formal defenses have costs**: Differential privacy provides provable guarantees but reduces model accuracy. Practitioners must navigate the privacy-utility trade-off carefully.

6. **Defense in depth is necessary**: Combine regularization, output perturbation, and architectural choices (like knowledge distillation) for robust protection.

7. **Audit your models**: Regularly run membership inference attacks against your own models to quantify privacy leakage before adversaries do.

8. **Consider the deployment context**: Models served with full probability outputs are far more vulnerable than those returning only hard predictions. Minimize the information exposed in model outputs.
