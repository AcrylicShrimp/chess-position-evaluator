use anyhow::Result;
use inference::Evaluator;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let model_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "models/onnx/small-0.6337-model-best.onnx".to_string()
    };

    println!("Loading model from: {}", model_path);
    let mut evaluator = Evaluator::from_path(&model_path)?;
    println!("Model loaded successfully!");

    println!("\nRunning inference with random input...");
    let batch_sizes = [1, 4, 16];

    for &batch_size in &batch_sizes {
        let results = evaluator.evaluate_random(batch_size)?;
        println!(
            "Batch size {}: {:?}",
            batch_size,
            results
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );
    }

    println!("\nONNX inference working!");
    Ok(())
}
