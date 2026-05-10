# CLIPDFExtraction

One-shot CLI that rasterizes PDFs page-by-page and runs `opendatalab/MinerU2.5-Pro-2604-1.2B` (via vLLM in-process) on each page, writing the extracted output to disk.

Designed to run inside the `VLLMMultimodal` Docker image. Mounts/configs assume that image's `run_configuration.yml`.

## Configurations

Two YAML files in `Configurations/` (copy `.example` → real names):

| File | Purpose |
| --- | --- |
| `mineru_configuration.yml` | model path + vLLM engine kwargs (dtype, max_model_len, gpu_memory_utilization, enforce_eager) |
| `pdf_extraction_configuration.yml` | input PDF path / dir, output dir, rasterization DPI, output image format, whether to retain intermediate page PNGs |

## Run

Inside the container:

```bash
cd /InServiceOfX/PythonApplications/CLIPDFExtraction

cp Configurations/mineru_configuration.yml.example   Configurations/mineru_configuration.yml
cp Configurations/pdf_extraction_configuration.yml.example \
   Configurations/pdf_extraction_configuration.yml
# edit the input/output paths

python Executables/main_CLIPDFExtraction.py --currentpath
```

`--currentpath` makes the app look for `Configurations/*.yml` in the current working directory (which is the app dir). `--configpath /some/dir` overrides to a specific dir, mirroring the `CLIImage` convention.

## Layout

```
clipdfextraction/
  ApplicationPaths.py            # path resolution + sys.path injection
  CLIPDFExtraction.py            # top-level orchestration
  Core/
    ProcessConfigurations.py     # loads YAML configs into pydantic objects
    PDFRasterizer.py             # PDF -> list[PIL.Image] via pdf2image
    ExtractionRunner.py          # iterates PDFs, calls MinerU, writes outputs
```

## Output layout

For an input PDF `foo.pdf`, output is written under `<output_dir>/foo/`:

```
<output_dir>/foo/
  page_1.md          # MinerU extraction (markdown-ish)
  page_1.png         # rasterized page (only if save_intermediate_images: true)
  page_2.md
  ...
  manifest.json      # {pdf_stem, num_pages, dpi, model_path, ...}
```
