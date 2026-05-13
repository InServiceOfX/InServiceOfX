# CLIPDFColQwenIndexer

CLI that rasterizes PDFs page-by-page, embeds each page image with
`vidore/colqwen2.5-v0.2`, and persists one multi-vector embedding file per page.

This is a retrieval indexer, not a parser. It does not extract labels or flow
paths directly. Use it to ask natural-language retrieval questions later with
`CLIPDFColQwenQuery`.

## Run

Inside the `VLLMMultimodal` container:

```bash
cd /InServiceOfX/PythonApplications/CLIPDFColQwenIndexer
cp Configurations/colqwen2_5_configuration.yml.example Configurations/colqwen2_5_configuration.yml
cp Configurations/pdf_index_configuration.yml.example Configurations/pdf_index_configuration.yml
python Executables/main_CLIPDFColQwenIndexer.py --currentpath
```

## Output

For an input PDF `foo.pdf`:

```text
<output_path>/foo/
  page_1.safetensors
  page_1.png
  page_2.safetensors
  ...
  manifest.json
```

Each `.safetensors` file contains:

- `embedding`: `[num_patches, embed_dim]`
- `page_index`: scalar tensor with the 1-based page index
