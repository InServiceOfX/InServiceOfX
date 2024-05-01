use candle_core::{Device, Result, Tensor};

#[test]
fn tensor_creates() -> Result<()>
{
	// To use GPU.
  let device = Device::new_cuda(0)?;

  // https://huggingface.github.io/candle/guide/cheatsheet.html
  let x = Tensor::new(&[[1f32, 2.], [3., 4.]], &device)?;

  //assert_eq!(x[0, 0]?, 1f32);

  assert_eq!(x.dims()[0], 2);
  //assert_eq!(x[0].dims(), 2);

  Ok(())
}

#[test]
fn matrix_multiplication_multiplies()
{
  assert!(true);
}