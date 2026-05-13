# CLIPDFColQwenQuery

CLI that embeds natural-language queries with `vidore/colqwen2.5-v0.2`, scores
them against a `CLIPDFColQwenIndexer` output directory, and writes top-K page
hits.

Use this for retrieval questions such as "which page shows nitrogen purge
valves?" or "which page has pressure-transducer manufacturer part numbers?".
It ranks pages; it does not generate a textual answer.

## Run

Inside the `VLLMMultimodal` container:

```bash
cd /InServiceOfX/PythonApplications/CLIPDFColQwenQuery
cp Configurations/colqwen2_5_configuration.yml.example Configurations/colqwen2_5_configuration.yml
cp Configurations/pdf_query_configuration.yml.example Configurations/pdf_query_configuration.yml
python Executables/main_CLIPDFColQwenQuery.py --currentpath
```

## Output

```json
{
  "queries": [
    {
      "query": "...",
      "hits": [
        {
          "rank": 1,
          "score": 16.12,
          "pdf": "sample-pid-document",
          "page": 8,
          "embedding": ".../page_8.safetensors",
          "page_image": ".../page_8.png"
        }
      ]
    }
  ]
}
```
