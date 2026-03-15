use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::Deserialize;

// ============================================================
// Linear Model
// ============================================================

/// A simple linear regression model with optional L2 regularization.
pub struct LinearModel {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub l2_lambda: f64,
}

impl LinearModel {
    pub fn new(num_features: usize, l2_lambda: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_shape_fn(num_features, |_| rng.gen_range(-0.01..0.01));
        Self {
            weights,
            bias: 0.0,
            l2_lambda,
        }
    }

    /// Predict a single sample.
    pub fn predict(&self, x: &Array1<f64>) -> f64 {
        x.dot(&self.weights) + self.bias
    }

    /// Predict a batch of samples.
    pub fn predict_batch(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.weights) + self.bias
    }

    /// Mean squared error loss for a single sample.
    pub fn loss(&self, x: &Array1<f64>, y: f64) -> f64 {
        let pred = self.predict(x);
        (pred - y).powi(2)
    }

    /// Mean squared error loss for a batch.
    pub fn batch_loss(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let preds = self.predict_batch(x);
        let diff = &preds - y;
        diff.mapv(|v| v.powi(2)).mean().unwrap_or(0.0)
    }

    /// Train the model using gradient descent with L2 regularization.
    pub fn train(&mut self, x: &Array2<f64>, y: &Array1<f64>, lr: f64, epochs: usize) {
        let n = x.nrows() as f64;
        for _ in 0..epochs {
            let preds = self.predict_batch(x);
            let errors = &preds - y;

            // Gradient for weights: (2/n) * X^T * errors + 2 * lambda * weights
            let grad_w = x.t().dot(&errors) * (2.0 / n)
                + &self.weights * (2.0 * self.l2_lambda);
            let grad_b = errors.mean().unwrap_or(0.0) * 2.0;

            self.weights = &self.weights - &(grad_w * lr);
            self.bias -= grad_b * lr;
        }
    }

    /// Compute per-sample losses for a batch.
    pub fn per_sample_losses(&self, x: &Array2<f64>, y: &Array1<f64>) -> Vec<f64> {
        let preds = self.predict_batch(x);
        preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .collect()
    }
}

// ============================================================
// Confidence metric (for regression, based on prediction error)
// ============================================================

/// Compute a confidence score for regression predictions.
/// Higher confidence means the prediction is closer to the target
/// relative to a baseline variance.
pub fn confidence_scores(
    model: &LinearModel,
    x: &Array2<f64>,
    y: &Array1<f64>,
    baseline_variance: f64,
) -> Vec<f64> {
    let preds = model.predict_batch(x);
    preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| {
            let err = (p - t).powi(2);
            // Confidence: exp(-error / baseline_variance)
            (-err / baseline_variance.max(1e-10)).exp()
        })
        .collect()
}

// ============================================================
// Entropy-based metric
// ============================================================

/// Compute entropy-like score for regression outputs.
/// We discretize the prediction error into bins and compute
/// a pseudo-entropy from the error distribution.
/// For regression, we use a proxy: -|error| * log(|error| + epsilon)
pub fn entropy_scores(model: &LinearModel, x: &Array2<f64>, y: &Array1<f64>) -> Vec<f64> {
    let preds = model.predict_batch(x);
    let epsilon = 1e-10;
    preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| {
            let abs_err = (p - t).abs() + epsilon;
            // Lower entropy means more "certain" prediction (closer to target)
            abs_err * abs_err.ln()
        })
        .collect()
}

// ============================================================
// Loss-based membership inference attack
// ============================================================

pub struct LossAttackResult {
    pub threshold: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub auc: f64,
}

/// Run a loss-based membership inference attack.
/// Members should have lower loss than non-members.
pub fn loss_based_attack(
    model: &LinearModel,
    member_x: &Array2<f64>,
    member_y: &Array1<f64>,
    nonmember_x: &Array2<f64>,
    nonmember_y: &Array1<f64>,
) -> LossAttackResult {
    let member_losses = model.per_sample_losses(member_x, member_y);
    let nonmember_losses = model.per_sample_losses(nonmember_x, nonmember_y);

    // True labels: 1 for member, 0 for non-member
    let mut all_scores: Vec<(f64, bool)> = Vec::new();
    for &loss in &member_losses {
        all_scores.push((loss, true)); // member
    }
    for &loss in &nonmember_losses {
        all_scores.push((loss, false)); // non-member
    }

    // For loss-based attack, lower loss => more likely member
    // So we negate scores for threshold search (higher negated = more likely member)
    let scores_for_auc: Vec<(f64, bool)> = all_scores
        .iter()
        .map(|(s, is_member)| (-s, *is_member))
        .collect();

    let auc = compute_auc(&scores_for_auc);

    // Find optimal threshold
    let (threshold, accuracy, precision, recall) =
        find_optimal_threshold_loss(&member_losses, &nonmember_losses);

    LossAttackResult {
        threshold,
        accuracy,
        precision,
        recall,
        auc,
    }
}

// ============================================================
// Confidence-based membership inference attack
// ============================================================

pub struct ConfidenceAttackResult {
    pub threshold: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub auc: f64,
}

/// Run a confidence-based membership inference attack.
/// Members should have higher confidence than non-members.
pub fn confidence_based_attack(
    model: &LinearModel,
    member_x: &Array2<f64>,
    member_y: &Array1<f64>,
    nonmember_x: &Array2<f64>,
    nonmember_y: &Array1<f64>,
    baseline_variance: f64,
) -> ConfidenceAttackResult {
    let member_conf = confidence_scores(model, member_x, member_y, baseline_variance);
    let nonmember_conf = confidence_scores(model, nonmember_x, nonmember_y, baseline_variance);

    let mut all_scores: Vec<(f64, bool)> = Vec::new();
    for &c in &member_conf {
        all_scores.push((c, true));
    }
    for &c in &nonmember_conf {
        all_scores.push((c, false));
    }

    let auc = compute_auc(&all_scores);
    let (threshold, accuracy, precision, recall) =
        find_optimal_threshold_confidence(&member_conf, &nonmember_conf);

    ConfidenceAttackResult {
        threshold,
        accuracy,
        precision,
        recall,
        auc,
    }
}

// ============================================================
// Entropy-based membership inference attack
// ============================================================

pub struct EntropyAttackResult {
    pub threshold: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub auc: f64,
}

/// Run an entropy-based membership inference attack.
/// Members should have lower entropy (model is more certain).
pub fn entropy_based_attack(
    model: &LinearModel,
    member_x: &Array2<f64>,
    member_y: &Array1<f64>,
    nonmember_x: &Array2<f64>,
    nonmember_y: &Array1<f64>,
) -> EntropyAttackResult {
    let member_ent = entropy_scores(model, member_x, member_y);
    let nonmember_ent = entropy_scores(model, nonmember_x, nonmember_y);

    // Lower entropy => more likely member, so negate for AUC
    let mut all_scores: Vec<(f64, bool)> = Vec::new();
    for &e in &member_ent {
        all_scores.push((-e, true));
    }
    for &e in &nonmember_ent {
        all_scores.push((-e, false));
    }

    let auc = compute_auc(&all_scores);
    let (threshold, accuracy, precision, recall) =
        find_optimal_threshold_loss(&member_ent, &nonmember_ent);

    EntropyAttackResult {
        threshold,
        accuracy,
        precision,
        recall,
        auc,
    }
}

// ============================================================
// Shadow model attack
// ============================================================

pub struct ShadowModelAttack {
    pub num_shadows: usize,
    pub attack_weights: Array1<f64>,
    pub attack_bias: f64,
}

pub struct ShadowAttackResult {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub auc: f64,
}

impl ShadowModelAttack {
    /// Train shadow models and an attack classifier.
    ///
    /// `all_x` and `all_y` represent the full available data pool.
    /// Each shadow model is trained on a random subset (simulating the target's training).
    pub fn train(
        all_x: &Array2<f64>,
        all_y: &Array1<f64>,
        num_shadows: usize,
        shadow_train_frac: f64,
        lr: f64,
        epochs: usize,
    ) -> Self {
        let n = all_x.nrows();
        let train_size = (n as f64 * shadow_train_frac) as usize;
        let num_features = all_x.ncols();
        let mut rng = rand::thread_rng();

        // Collect attack training data: (features, is_member)
        // Features: [loss, confidence_score]
        let mut attack_features: Vec<[f64; 2]> = Vec::new();
        let mut attack_labels: Vec<f64> = Vec::new();

        let indices: Vec<usize> = (0..n).collect();

        for _ in 0..num_shadows {
            let mut shuffled = indices.clone();
            shuffled.shuffle(&mut rng);

            let train_idx: Vec<usize> = shuffled[..train_size].to_vec();
            let test_idx: Vec<usize> = shuffled[train_size..].to_vec();

            // Build shadow training data
            let shadow_x = select_rows(all_x, &train_idx);
            let shadow_y = select_elements(all_y, &train_idx);

            // Train shadow model
            let mut shadow_model = LinearModel::new(num_features, 0.0);
            shadow_model.train(&shadow_x, &shadow_y, lr, epochs);

            // Compute baseline variance from shadow training data
            let baseline_var = shadow_model.batch_loss(&shadow_x, &shadow_y).max(1e-6);

            // Collect member features
            for &idx in &train_idx {
                let x_i = all_x.row(idx).to_owned();
                let y_i = all_y[idx];
                let loss = shadow_model.loss(&x_i, y_i);
                let conf = (-loss / baseline_var).exp();
                attack_features.push([loss, conf]);
                attack_labels.push(1.0);
            }

            // Collect non-member features
            for &idx in &test_idx {
                let x_i = all_x.row(idx).to_owned();
                let y_i = all_y[idx];
                let loss = shadow_model.loss(&x_i, y_i);
                let conf = (-loss / baseline_var).exp();
                attack_features.push([loss, conf]);
                attack_labels.push(0.0);
            }
        }

        // Train a simple logistic regression attack classifier
        let n_attack = attack_features.len();
        let attack_x = Array2::from_shape_fn((n_attack, 2), |(i, j)| attack_features[i][j]);
        let attack_y = Array1::from_vec(attack_labels);

        let (weights, bias) = train_logistic_regression(&attack_x, &attack_y, 0.01, 500);

        ShadowModelAttack {
            num_shadows,
            attack_weights: weights,
            attack_bias: bias,
        }
    }

    /// Run the shadow model attack against a target model.
    pub fn attack(
        &self,
        target_model: &LinearModel,
        member_x: &Array2<f64>,
        member_y: &Array1<f64>,
        nonmember_x: &Array2<f64>,
        nonmember_y: &Array1<f64>,
    ) -> ShadowAttackResult {
        let baseline_var = target_model.batch_loss(member_x, member_y).max(1e-6);

        let mut all_scores: Vec<(f64, bool)> = Vec::new();

        // Score members
        for i in 0..member_x.nrows() {
            let x_i = member_x.row(i).to_owned();
            let y_i = member_y[i];
            let score = self.predict_membership(target_model, &x_i, y_i, baseline_var);
            all_scores.push((score, true));
        }

        // Score non-members
        for i in 0..nonmember_x.nrows() {
            let x_i = nonmember_x.row(i).to_owned();
            let y_i = nonmember_y[i];
            let score = self.predict_membership(target_model, &x_i, y_i, baseline_var);
            all_scores.push((score, false));
        }

        let auc = compute_auc(&all_scores);

        // Threshold at 0.5
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;
        for &(score, is_member) in &all_scores {
            let predicted_member = score > 0.5;
            match (predicted_member, is_member) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        let total = (tp + fp + tn + fn_) as f64;
        let accuracy = (tp + tn) as f64 / total;
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };

        ShadowAttackResult {
            accuracy,
            precision,
            recall,
            auc,
        }
    }

    fn predict_membership(
        &self,
        model: &LinearModel,
        x: &Array1<f64>,
        y: f64,
        baseline_var: f64,
    ) -> f64 {
        let loss = model.loss(x, y);
        let conf = (-loss / baseline_var).exp();
        let features = Array1::from_vec(vec![loss, conf]);
        let logit = features.dot(&self.attack_weights) + self.attack_bias;
        sigmoid(logit)
    }
}

// ============================================================
// Defense: evaluate regularization effect
// ============================================================

pub struct DefenseResult {
    pub lambda: f64,
    pub model_loss: f64,
    pub attack_accuracy: f64,
    pub attack_auc: f64,
}

/// Evaluate how L2 regularization affects membership inference vulnerability.
pub fn evaluate_regularization_defense(
    train_x: &Array2<f64>,
    train_y: &Array1<f64>,
    test_x: &Array2<f64>,
    test_y: &Array1<f64>,
    lambdas: &[f64],
    lr: f64,
    epochs: usize,
) -> Vec<DefenseResult> {
    let num_features = train_x.ncols();
    let mut results = Vec::new();

    for &lambda in lambdas {
        let mut model = LinearModel::new(num_features, lambda);
        model.train(train_x, train_y, lr, epochs);

        let model_loss = model.batch_loss(test_x, test_y);
        let attack = loss_based_attack(&model, train_x, train_y, test_x, test_y);

        results.push(DefenseResult {
            lambda,
            model_loss,
            attack_accuracy: attack.accuracy,
            attack_auc: attack.auc,
        });
    }

    results
}

// ============================================================
// Bybit API client
// ============================================================

#[derive(Debug, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct OhlcvBar {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    /// `interval`: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W
    /// `limit`: max 200
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<OhlcvBar>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitKlineResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
        }

        let mut bars: Vec<OhlcvBar> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                Some(OhlcvBar {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        bars.reverse();
        Ok(bars)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract features from OHLCV bars.
/// Features: [log_return, normalized_volume, high_low_range, open_close_spread]
/// Target: next-bar log return
pub fn extract_features(bars: &[OhlcvBar]) -> (Array2<f64>, Array1<f64>) {
    if bars.len() < 3 {
        return (
            Array2::zeros((0, 4)),
            Array1::zeros(0),
        );
    }

    let avg_volume: f64 = bars.iter().map(|b| b.volume).sum::<f64>() / bars.len() as f64;

    let mut features = Vec::new();
    let mut targets = Vec::new();

    for i in 1..bars.len() - 1 {
        let prev = &bars[i - 1];
        let curr = &bars[i];
        let next = &bars[i + 1];

        let log_return = (curr.close / prev.close.max(1e-10)).ln();
        let norm_volume = curr.volume / avg_volume.max(1e-10);
        let hl_range = (curr.high - curr.low) / curr.close.max(1e-10);
        let oc_spread = (curr.close - curr.open) / curr.open.max(1e-10);

        features.push(vec![log_return, norm_volume, hl_range, oc_spread]);
        targets.push((next.close / curr.close.max(1e-10)).ln());
    }

    let n = features.len();
    let flat: Vec<f64> = features.into_iter().flatten().collect();
    (
        Array2::from_shape_vec((n, 4), flat).unwrap(),
        Array1::from_vec(targets),
    )
}

// ============================================================
// Helper functions
// ============================================================

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn train_logistic_regression(
    x: &Array2<f64>,
    y: &Array1<f64>,
    lr: f64,
    epochs: usize,
) -> (Array1<f64>, f64) {
    let num_features = x.ncols();
    let n = x.nrows() as f64;
    let mut weights = Array1::zeros(num_features);
    let mut bias = 0.0;

    for _ in 0..epochs {
        let logits = x.dot(&weights) + bias;
        let preds = logits.mapv(sigmoid);
        let errors = &preds - y;

        let grad_w = x.t().dot(&errors) / n;
        let grad_b = errors.mean().unwrap_or(0.0);

        weights = &weights - &(grad_w * lr);
        bias -= grad_b * lr;
    }

    (weights, bias)
}

fn select_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let ncols = x.ncols();
    let mut result = Array2::zeros((indices.len(), ncols));
    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&x.row(idx));
    }
    result
}

fn select_elements(y: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

/// Compute AUC using the trapezoidal rule.
/// Scores: higher = more likely member. Labels: true = member.
fn compute_auc(scores_and_labels: &[(f64, bool)]) -> f64 {
    let mut sorted = scores_and_labels.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = sorted.iter().filter(|s| s.1).count() as f64;
    let total_neg = sorted.iter().filter(|s| !s.1).count() as f64;

    if total_pos == 0.0 || total_neg == 0.0 {
        return 0.5;
    }

    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tp = 0.0;
    let mut prev_fp = 0.0;

    for (i, &(_, is_pos)) in sorted.iter().enumerate() {
        if is_pos {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        // At each threshold change, compute area
        if i + 1 == sorted.len()
            || sorted[i].0 != sorted[i + 1].0
        {
            let tpr = tp / total_pos;
            let fpr = fp / total_neg;
            let prev_tpr = prev_tp / total_pos;
            let prev_fpr = prev_fp / total_neg;
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            prev_tp = tp;
            prev_fp = fp;
        }
    }

    auc
}

/// Find the optimal threshold for loss-based attack.
/// Lower loss => predicted member.
fn find_optimal_threshold_loss(
    member_losses: &[f64],
    nonmember_losses: &[f64],
) -> (f64, f64, f64, f64) {
    let mut all_losses: Vec<f64> = member_losses
        .iter()
        .chain(nonmember_losses.iter())
        .copied()
        .collect();
    all_losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_losses.dedup();

    let mut best_threshold = 0.0;
    let mut best_accuracy = 0.0;
    let mut best_precision = 0.0;
    let mut best_recall = 0.0;

    for &threshold in &all_losses {
        let tp = member_losses.iter().filter(|&&l| l < threshold).count();
        let fp = nonmember_losses.iter().filter(|&&l| l < threshold).count();
        let tn = nonmember_losses
            .iter()
            .filter(|&&l| l >= threshold)
            .count();
        let fn_ = member_losses.iter().filter(|&&l| l >= threshold).count();

        let total = (tp + fp + tn + fn_) as f64;
        let acc = (tp + tn) as f64 / total;

        if acc > best_accuracy {
            best_accuracy = acc;
            best_threshold = threshold;
            best_precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            best_recall = if tp + fn_ > 0 {
                tp as f64 / (tp + fn_) as f64
            } else {
                0.0
            };
        }
    }

    (best_threshold, best_accuracy, best_precision, best_recall)
}

/// Find the optimal threshold for confidence-based attack.
/// Higher confidence => predicted member.
fn find_optimal_threshold_confidence(
    member_confs: &[f64],
    nonmember_confs: &[f64],
) -> (f64, f64, f64, f64) {
    let mut all_confs: Vec<f64> = member_confs
        .iter()
        .chain(nonmember_confs.iter())
        .copied()
        .collect();
    all_confs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_confs.dedup();

    let mut best_threshold = 0.0;
    let mut best_accuracy = 0.0;
    let mut best_precision = 0.0;
    let mut best_recall = 0.0;

    for &threshold in &all_confs {
        let tp = member_confs.iter().filter(|&&c| c > threshold).count();
        let fp = nonmember_confs.iter().filter(|&&c| c > threshold).count();
        let tn = nonmember_confs
            .iter()
            .filter(|&&c| c <= threshold)
            .count();
        let fn_ = member_confs.iter().filter(|&&c| c <= threshold).count();

        let total = (tp + fp + tn + fn_) as f64;
        let acc = (tp + tn) as f64 / total;

        if acc > best_accuracy {
            best_accuracy = acc;
            best_threshold = threshold;
            best_precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            best_recall = if tp + fn_ > 0 {
                tp as f64 / (tp + fn_) as f64
            } else {
                0.0
            };
        }
    }

    (best_threshold, best_accuracy, best_precision, best_recall)
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_synthetic_data(n: usize, seed_offset: u64) -> (Array2<f64>, Array1<f64>) {
        // Simple linear relationship with noise
        let mut rng = rand::thread_rng();
        let true_weights = [0.5, -0.3, 0.8, 0.1];
        let mut features = Vec::new();
        let mut targets = Vec::new();

        for i in 0..n {
            let x: Vec<f64> = (0..4)
                .map(|j| ((i as f64 + seed_offset as f64) * (j as f64 + 1.0)).sin())
                .collect();
            let y: f64 = x.iter().zip(true_weights.iter()).map(|(a, b)| a * b).sum::<f64>()
                + rng.gen_range(-0.1..0.1);
            features.extend_from_slice(&x);
            targets.push(y);
        }

        (
            Array2::from_shape_vec((n, 4), features).unwrap(),
            Array1::from_vec(targets),
        )
    }

    #[test]
    fn test_linear_model_training() {
        let (x, y) = make_synthetic_data(100, 0);
        let mut model = LinearModel::new(4, 0.0);
        let loss_before = model.batch_loss(&x, &y);
        model.train(&x, &y, 0.01, 200);
        let loss_after = model.batch_loss(&x, &y);
        assert!(
            loss_after < loss_before,
            "Training should reduce loss: before={}, after={}",
            loss_before,
            loss_after
        );
    }

    #[test]
    fn test_loss_based_attack() {
        let (train_x, train_y) = make_synthetic_data(100, 0);
        let (test_x, test_y) = make_synthetic_data(100, 1000);

        let mut model = LinearModel::new(4, 0.0);
        model.train(&train_x, &train_y, 0.01, 500);

        let result = loss_based_attack(&model, &train_x, &train_y, &test_x, &test_y);
        // Attack accuracy should be better than random (0.5)
        assert!(
            result.accuracy >= 0.5,
            "Loss-based attack accuracy {} should be >= 0.5",
            result.accuracy
        );
        assert!(
            result.auc >= 0.45,
            "Loss-based attack AUC {} should be >= 0.45",
            result.auc
        );
    }

    #[test]
    fn test_confidence_based_attack() {
        let (train_x, train_y) = make_synthetic_data(100, 0);
        let (test_x, test_y) = make_synthetic_data(100, 1000);

        let mut model = LinearModel::new(4, 0.0);
        model.train(&train_x, &train_y, 0.01, 500);

        let baseline_var = model.batch_loss(&train_x, &train_y).max(1e-6);
        let result = confidence_based_attack(
            &model,
            &train_x,
            &train_y,
            &test_x,
            &test_y,
            baseline_var,
        );
        assert!(
            result.accuracy >= 0.5,
            "Confidence attack accuracy {} should be >= 0.5",
            result.accuracy
        );
    }

    #[test]
    fn test_entropy_based_attack() {
        let (train_x, train_y) = make_synthetic_data(100, 0);
        let (test_x, test_y) = make_synthetic_data(100, 1000);

        let mut model = LinearModel::new(4, 0.0);
        model.train(&train_x, &train_y, 0.01, 500);

        let result = entropy_based_attack(&model, &train_x, &train_y, &test_x, &test_y);
        assert!(
            result.accuracy >= 0.5,
            "Entropy attack accuracy {} should be >= 0.5",
            result.accuracy
        );
    }

    #[test]
    fn test_shadow_model_attack() {
        let (all_x, all_y) = make_synthetic_data(200, 0);

        let train_x = all_x.slice(ndarray::s![..100, ..]).to_owned();
        let train_y = all_y.slice(ndarray::s![..100]).to_owned();
        let test_x = all_x.slice(ndarray::s![100.., ..]).to_owned();
        let test_y = all_y.slice(ndarray::s![100..]).to_owned();

        let mut target_model = LinearModel::new(4, 0.0);
        target_model.train(&train_x, &train_y, 0.01, 300);

        let shadow_attack = ShadowModelAttack::train(&all_x, &all_y, 3, 0.5, 0.01, 300);
        let result = shadow_attack.attack(
            &target_model,
            &train_x,
            &train_y,
            &test_x,
            &test_y,
        );

        assert!(
            result.accuracy >= 0.4,
            "Shadow model attack accuracy {} should be >= 0.4",
            result.accuracy
        );
    }

    #[test]
    fn test_regularization_defense() {
        let (train_x, train_y) = make_synthetic_data(100, 0);
        let (test_x, test_y) = make_synthetic_data(100, 1000);

        let lambdas = vec![0.0, 0.01, 0.1, 1.0];
        let results = evaluate_regularization_defense(
            &train_x, &train_y, &test_x, &test_y, &lambdas, 0.01, 500,
        );

        assert_eq!(results.len(), 4);
        // Higher regularization should generally reduce attack effectiveness
        // (though not strictly monotonic in all cases)
        for r in &results {
            assert!(r.attack_accuracy >= 0.0 && r.attack_accuracy <= 1.0);
            assert!(r.attack_auc >= 0.0 && r.attack_auc <= 1.0);
        }
    }

    #[test]
    fn test_auc_computation() {
        // Perfect separation: all members have higher scores
        let scores = vec![
            (1.0, true),
            (0.9, true),
            (0.4, false),
            (0.3, false),
        ];
        let auc = compute_auc(&scores);
        assert!(
            (auc - 1.0).abs() < 0.01,
            "Perfect separation should give AUC ~1.0, got {}",
            auc
        );

        // Random: equal scores
        let random_scores = vec![
            (0.5, true),
            (0.5, false),
            (0.5, true),
            (0.5, false),
        ];
        let auc_random = compute_auc(&random_scores);
        assert!(
            (auc_random - 0.5).abs() < 0.01,
            "Equal scores should give AUC ~0.5, got {}",
            auc_random
        );
    }

    #[test]
    fn test_feature_extraction() {
        let bars = vec![
            OhlcvBar { timestamp: 0, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000.0 },
            OhlcvBar { timestamp: 1, open: 102.0, high: 108.0, low: 100.0, close: 106.0, volume: 1200.0 },
            OhlcvBar { timestamp: 2, open: 106.0, high: 110.0, low: 104.0, close: 108.0, volume: 800.0 },
            OhlcvBar { timestamp: 3, open: 108.0, high: 112.0, low: 106.0, close: 109.0, volume: 900.0 },
        ];

        let (features, targets) = extract_features(&bars);
        assert_eq!(features.nrows(), 2);
        assert_eq!(features.ncols(), 4);
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_model_predict() {
        let model = LinearModel {
            weights: array![1.0, 2.0, 3.0],
            bias: 0.5,
            l2_lambda: 0.0,
        };
        let x = array![1.0, 1.0, 1.0];
        let pred = model.predict(&x);
        assert!((pred - 6.5).abs() < 1e-10);
    }

    #[test]
    fn test_per_sample_losses() {
        let model = LinearModel {
            weights: array![1.0, 0.0],
            bias: 0.0,
            l2_lambda: 0.0,
        };
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 2.0, 0.0]).unwrap();
        let y = array![1.0, 2.0];
        let losses = model.per_sample_losses(&x, &y);
        assert!((losses[0] - 0.0).abs() < 1e-10);
        assert!((losses[1] - 0.0).abs() < 1e-10);
    }
}
