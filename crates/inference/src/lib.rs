use anyhow::Result;
use ndarray::{Array, Array4};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::path::Path;

/// Chess position evaluator using ONNX runtime.
pub struct Evaluator {
    session: Session,
}

impl Evaluator {
    /// Load an ONNX model from the given path.
    pub fn from_path(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Run inference on a batch of encoded boards.
    ///
    /// Input shape: [batch, 18, 8, 8]
    /// Output: Vec of win probabilities (sigmoid applied)
    pub fn evaluate_batch(&mut self, boards: Array4<f32>) -> Result<Vec<f32>> {
        let input = TensorRef::from_array_view(&boards)?;
        let outputs = self.session.run(ort::inputs!["board" => input])?;

        let (_shape, data) = outputs["value"].try_extract_tensor::<f32>()?;
        let values: Vec<f32> = data.iter().map(|&v| sigmoid(v)).collect();

        Ok(values)
    }

    /// Run inference on a single encoded board.
    ///
    /// Input shape: [18, 8, 8]
    /// Output: win probability (0.0 to 1.0)
    pub fn evaluate_single(&mut self, board: &Array<f32, ndarray::Ix3>) -> Result<f32> {
        let batch = board.clone().insert_axis(ndarray::Axis(0));
        let results = self.evaluate_batch(batch)?;
        Ok(results[0])
    }

    /// Evaluate with random input (for testing).
    pub fn evaluate_random(&mut self, batch_size: usize) -> Result<Vec<f32>> {
        use ndarray::Array;
        let random_input: Array4<f32> =
            Array::from_shape_fn((batch_size, 18, 8, 8), |_| rand_f32());
        self.evaluate_batch(random_input)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Simple pseudo-random f32 in [0, 1) for testing.
fn rand_f32() -> f32 {
    use std::time::SystemTime;
    static mut SEED: u64 = 0;
    unsafe {
        if SEED == 0 {
            SEED = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
        }
        // xorshift64
        SEED ^= SEED << 13;
        SEED ^= SEED >> 7;
        SEED ^= SEED << 17;
        (SEED as f32) / (u64::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
