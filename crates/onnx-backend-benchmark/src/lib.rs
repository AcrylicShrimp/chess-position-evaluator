use anyhow::Result;
use ndarray::{Array, Array4};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::path::Path;

/// Default model path for benchmarks.
pub const DEFAULT_MODEL_PATH: &str = "models/onnx/small-0.6337-model-best.onnx";

/// Input tensor shape constants.
pub const CHANNELS: usize = 18;
pub const BOARD_SIZE: usize = 8;

/// Create an ONNX session with full optimization.
pub fn create_session(model_path: impl AsRef<Path>) -> Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .commit_from_file(model_path)?;
    Ok(session)
}

/// Generate random input tensor for benchmarking.
pub fn generate_random_input(batch_size: usize) -> Array4<f32> {
    Array::from_shape_fn((batch_size, CHANNELS, BOARD_SIZE, BOARD_SIZE), |_| {
        rand_f32()
    })
}

/// Run inference on a batch of inputs.
pub fn run_inference(session: &mut Session, input: &Array4<f32>) -> Result<Vec<f32>> {
    let tensor = TensorRef::from_array_view(input)?;
    let outputs = session.run(ort::inputs!["board" => tensor])?;
    let (_shape, data) = outputs["value"].try_extract_tensor::<f32>()?;
    Ok(data.iter().map(|&v| sigmoid(v)).collect())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Simple pseudo-random f32 in [0, 1) for benchmarking.
fn rand_f32() -> f32 {
    use std::cell::Cell;
    use std::time::SystemTime;

    thread_local! {
        static SEED: Cell<u64> = Cell::new(0);
    }

    SEED.with(|seed| {
        let mut s = seed.get();
        if s == 0 {
            s = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
        }
        // xorshift64
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        seed.set(s);
        (s as f32) / (u64::MAX as f32)
    })
}
