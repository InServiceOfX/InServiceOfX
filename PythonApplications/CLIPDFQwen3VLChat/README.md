# CLIPDFQwen3VLChat

CLI that rasterizes PDFs and feeds each page to `Qwen3-VL-4B-Instruct` (default: the `cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit` quant) with a user-supplied prompt. One freeform `page_N.txt` per page plus a `manifest.json` per PDF.

Mirrors `CLIPDFExtraction`'s structure but emits **unstructured natural-language responses** (not MinerU's per-element JSON). Use this when you want the model to *answer a question about* each page rather than extract its layout.

Designed to run inside the `VLLMMultimodal` Docker image (the same image used by `CLIPDFExtraction`).

## Configurations

Two YAML files in `Configurations/` (copy `.example` → real names):

- `qwen3vl_configuration.yml` — Qwen3-VL model + vLLM engine kwargs (model_path, quantization, max_model_len, sampling overrides)
- `pdf_chat_configuration.yml` — input PDF(s), output dir, prompt to send to the model, DPI, etc.

## Run

```bash
# inside the container:
cd /InServiceOfX/PythonApplications/CLIPDFQwen3VLChat
cp Configurations/qwen3vl_configuration.yml.example       Configurations/qwen3vl_configuration.yml
cp Configurations/pdf_chat_configuration.yml.example      Configurations/pdf_chat_configuration.yml
# edit pdf_chat_configuration.yml to set the prompt and input_path

python Executables/main_CLIPDFQwen3VLChat.py --currentpath
```

## Output layout

```
<output_path>/<pdf_stem>/
  page_1.txt           # Qwen3-VL freeform response for page 1
  page_1.png           # (if save_intermediate_images: true) rasterized source
  ...
  manifest.json        # {pdf_path, num_pages, dpi, prompt, pages: [...]}
```

`skip_existing: true` makes re-runs idempotent (skips pages whose `.txt` already exists).
