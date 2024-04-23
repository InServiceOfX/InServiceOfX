Because we are running PyTorch 2.2 or higher, we do not, for HuggingFace's diffusers need to use any extra dependencies such as xFormers. So we don't have to do this in our Dockerfile:

```
## xFormers - optimizations performed in attention blocks; recommended by
  # huggingface.
  # See https://huggingface.co/docs/diffusers/en/optimization/xformers
  pip install --upgrade xformers && \

```

See https://huggingface.co/docs/diffusers/en/optimization/torch2.0