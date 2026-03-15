use membership_inference::*;

fn main() -> anyhow::Result<()> {
    println!("=== Chapter 230: Membership Inference in Trading ===\n");

    // --------------------------------------------------------
    // Step 1: Fetch data from Bybit
    // --------------------------------------------------------
    println!("[1] Fetching BTCUSDT kline data from Bybit...");
    let client = BybitClient::new();
    let bars = match client.fetch_klines("BTCUSDT", "60", 200) {
        Ok(b) if b.len() >= 10 => {
            println!("    Fetched {} bars from Bybit.", b.len());
            b
        }
        Ok(b) => {
            println!(
                "    Only got {} bars from Bybit, generating synthetic data instead.",
                b.len()
            );
            generate_synthetic_bars(200)
        }
        Err(e) => {
            println!("    Bybit API error: {}. Using synthetic data.", e);
            generate_synthetic_bars(200)
        }
    };

    // --------------------------------------------------------
    // Step 2: Extract features
    // --------------------------------------------------------
    println!("\n[2] Extracting features from OHLCV data...");
    let (features, targets) = extract_features(&bars);
    let n = features.nrows();
    println!("    Total samples: {}, Features per sample: {}", n, features.ncols());

    if n < 20 {
        println!("    Not enough data points. Exiting.");
        return Ok(());
    }

    // --------------------------------------------------------
    // Step 3: Split into member / non-member sets
    // --------------------------------------------------------
    let split = n / 2;
    let member_x = features.slice(ndarray::s![..split, ..]).to_owned();
    let member_y = targets.slice(ndarray::s![..split]).to_owned();
    let nonmember_x = features.slice(ndarray::s![split.., ..]).to_owned();
    let nonmember_y = targets.slice(ndarray::s![split..]).to_owned();
    println!(
        "\n[3] Data split: {} members, {} non-members",
        member_x.nrows(),
        nonmember_x.nrows()
    );

    // --------------------------------------------------------
    // Step 4: Train target model (no regularization)
    // --------------------------------------------------------
    println!("\n[4] Training target model (no regularization)...");
    let mut target_model = LinearModel::new(4, 0.0);
    target_model.train(&member_x, &member_y, 0.005, 1000);
    let train_loss = target_model.batch_loss(&member_x, &member_y);
    let test_loss = target_model.batch_loss(&nonmember_x, &nonmember_y);
    println!("    Training loss: {:.6}", train_loss);
    println!("    Test loss:     {:.6}", test_loss);
    println!("    Generalization gap: {:.6}", test_loss - train_loss);

    // --------------------------------------------------------
    // Step 5: Run membership inference attacks
    // --------------------------------------------------------
    println!("\n[5] Running membership inference attacks...\n");

    // 5a. Loss-based attack
    let loss_result = loss_based_attack(
        &target_model,
        &member_x,
        &member_y,
        &nonmember_x,
        &nonmember_y,
    );
    println!("--- Loss-Based Attack ---");
    println!("  Threshold: {:.6}", loss_result.threshold);
    println!("  Accuracy:  {:.4}", loss_result.accuracy);
    println!("  Precision: {:.4}", loss_result.precision);
    println!("  Recall:    {:.4}", loss_result.recall);
    println!("  AUC:       {:.4}", loss_result.auc);

    // 5b. Confidence-based attack
    let baseline_var = target_model.batch_loss(&member_x, &member_y).max(1e-6);
    let conf_result = confidence_based_attack(
        &target_model,
        &member_x,
        &member_y,
        &nonmember_x,
        &nonmember_y,
        baseline_var,
    );
    println!("\n--- Confidence-Based Attack ---");
    println!("  Threshold: {:.6}", conf_result.threshold);
    println!("  Accuracy:  {:.4}", conf_result.accuracy);
    println!("  Precision: {:.4}", conf_result.precision);
    println!("  Recall:    {:.4}", conf_result.recall);
    println!("  AUC:       {:.4}", conf_result.auc);

    // 5c. Entropy-based attack
    let ent_result = entropy_based_attack(
        &target_model,
        &member_x,
        &member_y,
        &nonmember_x,
        &nonmember_y,
    );
    println!("\n--- Entropy-Based Attack ---");
    println!("  Threshold: {:.6}", ent_result.threshold);
    println!("  Accuracy:  {:.4}", ent_result.accuracy);
    println!("  Precision: {:.4}", ent_result.precision);
    println!("  Recall:    {:.4}", ent_result.recall);
    println!("  AUC:       {:.4}", ent_result.auc);

    // 5d. Shadow model attack
    println!("\n--- Shadow Model Attack ---");
    println!("  Training 5 shadow models...");
    let shadow_attack =
        ShadowModelAttack::train(&features, &targets, 5, 0.5, 0.005, 500);
    let shadow_result = shadow_attack.attack(
        &target_model,
        &member_x,
        &member_y,
        &nonmember_x,
        &nonmember_y,
    );
    println!("  Accuracy:  {:.4}", shadow_result.accuracy);
    println!("  Precision: {:.4}", shadow_result.precision);
    println!("  Recall:    {:.4}", shadow_result.recall);
    println!("  AUC:       {:.4}", shadow_result.auc);

    // --------------------------------------------------------
    // Step 6: Defense - L2 Regularization
    // --------------------------------------------------------
    println!("\n[6] Evaluating L2 regularization defense...\n");
    let lambdas = vec![0.0, 0.001, 0.01, 0.1, 1.0];
    let defense_results = evaluate_regularization_defense(
        &member_x,
        &member_y,
        &nonmember_x,
        &nonmember_y,
        &lambdas,
        0.005,
        1000,
    );

    println!(
        "{:<12} {:<14} {:<16} {:<10}",
        "Lambda", "Model Loss", "Attack Accuracy", "Attack AUC"
    );
    println!("{}", "-".repeat(52));
    for r in &defense_results {
        println!(
            "{:<12.4} {:<14.6} {:<16.4} {:<10.4}",
            r.lambda, r.model_loss, r.attack_accuracy, r.attack_auc
        );
    }

    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------
    println!("\n=== Summary ===");
    println!(
        "Best attack (by AUC): Loss={:.4}, Conf={:.4}, Entropy={:.4}, Shadow={:.4}",
        loss_result.auc, conf_result.auc, ent_result.auc, shadow_result.auc
    );

    if defense_results.len() >= 2 {
        let no_reg = &defense_results[0];
        let strong_reg = defense_results.last().unwrap();
        println!(
            "Regularization effect: AUC dropped from {:.4} to {:.4} (lambda {} -> {})",
            no_reg.attack_auc, strong_reg.attack_auc, no_reg.lambda, strong_reg.lambda
        );
    }

    println!("\nDone.");
    Ok(())
}

/// Generate synthetic OHLCV bars as fallback when Bybit API is unavailable.
fn generate_synthetic_bars(n: usize) -> Vec<OhlcvBar> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut bars = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        let ret = rng.gen_range(-0.02..0.02);
        let close = price * (1.0 + ret);
        let high = close * (1.0 + rng.gen_range(0.0..0.01));
        let low = close * (1.0 - rng.gen_range(0.0..0.01));
        let open = price * (1.0 + rng.gen_range(-0.005..0.005));
        let volume = rng.gen_range(100.0..10000.0);

        bars.push(OhlcvBar {
            timestamp: (1700000000 + i * 3600) as u64,
            open,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }
    bars
}
