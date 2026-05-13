# PaddleOCRVL API Wrapper

Thin wrapper for calling a local OpenAI-compatible vLLM server running
`PaddlePaddle/PaddleOCR-VL-1.5`.

This library intentionally lives under `ThirdParties/APIs`, not
`HuggingFace`, because it does not use `transformers` directly. It sends image
pages to a vLLM HTTP server and returns the generated text.

Start the server inside `vllm-multimodal:25.06-py3`:

```bash
vllm serve /Data/Models/Multimodal/PaddlePaddle/PaddleOCR-VL-1.5 \
  --served-model-name paddleocr-vl \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8080 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

Then call `http://127.0.0.1:8080/v1/chat/completions`.

This is not the full official PaddleOCR `PaddleOCRVL` pipeline. It is the
direct VLM-recognition path for comparison against MinerU and Qwen3-VL.
