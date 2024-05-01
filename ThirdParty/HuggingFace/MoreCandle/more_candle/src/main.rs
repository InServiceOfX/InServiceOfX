use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

struct Model
{
  first: Tensor,
  second: Tensor,
}

impl Model
{
  fn forward(&self, image: &Tensor) -> Result<Tensor>
  {
    let x = image.matmul(&self.first)?;
    let x = x.relu()?;
    x.matmul(&self.second)
  }
}

// Using a Linear layer.
// Add bias and create a classical Linear layer.
struct LinearExample
{
  weight: Tensor,
  bias: Tensor,
}

impl LinearExample
{
  fn forward(&self, x: &Tensor) -> Result<Tensor>
  {
    let x = x.matmul(&self.weight)?;
    x.broadcast_add(&self.bias)
  }
}

struct ModelWithLinearLayers
{
  first: LinearExample,
  second: LinearExample,
}

impl ModelWithLinearLayers
{
  fn forward(&self, image: &Tensor) -> Result<Tensor>
  {
    let x = self.first.forward(image)?;
    let x = x.relu()?;
    self.second.forward(&x)
  }
}

struct ModelWithNN
{
  first: Linear,
  second: Linear,
}

impl ModelWithNN
{
  fn forward(&self, image: &Tensor) -> Result<Tensor>
  {
    let x = self.first.forward(image)?;
    let x = x.relu()?;
    self.second.forward(&x)
  }
}

fn main() -> Result<()>
{
  println!("Hello, world! This is for candles");

  // To use GPU.
  let device = Device::new_cuda(0)?;

  let first = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
  let second = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
  let model = Model { first, second };

  let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

  let digit = model.forward(&dummy_image)?;
  println!("Digit {digit:?} digit");

  // Using a Linear layer (example).
  let device = Device::cuda_if_available(0)?;

  // Creating a dummy model
  let weight = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
  let bias = Tensor::randn(0f32, 1.0, (100,), &device)?;
  let first = LinearExample{weight, bias};
  let weight = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
  let bias = Tensor::randn(0f32, 1.0, (10,), &device)?;
  let second = LinearExample{weight, bias};
  let model = ModelWithLinearLayers{first, second};

  let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

  // Inference on the model
  let digit = model.forward(&dummy_image)?;
  println!("From custom examples: Digit {digit:?} digit");

  // This has changed (784, 100) -> (100, 784)!
  // This is because Linear is coded with the PyTorch layout in mind, so it uses
  // the transpose of the weights and not the weights directly.

  let weight = Tensor::randn(0f32, 1.0, (100, 784), &device)?;
  let bias = Tensor::randn(0f32, 1.0, (100,), &device)?;
  let first = Linear::new(weight, Some(bias));
  let weight = Tensor::randn(0f32, 1.0, (10, 100), &device)?;
  let bias = Tensor::randn(0f32, 1.0, (10,), &device)?;
  let second = Linear::new(weight, Some(bias));
  let model = ModelWithNN{first, second};

  let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

  let digit = model.forward(&dummy_image)?;
  println!("With nn, Digit {digit:?} digit");

  Ok(())
}
