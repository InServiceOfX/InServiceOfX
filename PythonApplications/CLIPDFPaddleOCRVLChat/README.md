# CLIPDFPaddleOCRVLChat

PDF page runner for `PaddlePaddle/PaddleOCR-VL-1.5` served by vLLM.

This app is for comparing direct PaddleOCR-VL recognition against MinerU and
Qwen3-VL on the same P&ID PDF. It calls a running OpenAI-compatible vLLM server
through `PythonLibraries/ThirdParties/APIs/PaddleOCRVL`.

## Start The Server

Inside `vllm-multimodal:25.06-py3`:

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

The current image can load this model with those caps. The default 131k context
is too heavy for a quick 12 GB consumer-GPU workflow.

## Run The PDF Client

In another shell/container with access to the same server:

```bash
cd /InServiceOfX/PythonApplications/CLIPDFPaddleOCRVLChat
cp Configurations/paddleocrvl_api_configuration.yml.example Configurations/paddleocrvl_api_configuration.yml
cp Configurations/pdf_chat_configuration.yml.example Configurations/pdf_chat_configuration.yml
python Executables/main_CLIPDFPaddleOCRVLChat.py --currentpath
```

Outputs:

```text
<output_path>/<pdf_stem>/
  page_1.txt
  page_1.png
  manifest.json
```
