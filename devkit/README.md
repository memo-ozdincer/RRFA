# RRFA Dev Kit

Local visualization tool for the data pipeline. Run on Mac, no GPU needed.

```bash
pip install streamlit pandas transformers
streamlit run devkit/app.py
```

## Pages

- **Dashboard** - File counts, per-dataset label distributions, sequence length stats
- **Trace Explorer** - Browse traces with color-coded messages, injection highlighting, tool call details
- **Render & Lossmask** - Live ETL_B rendering with token-level loss mask heatmap (needs Llama tokenizer)
- **Policy Comparator** - All 8 LMP policies side-by-side on one trace
- **Dataset Importer** - Test ETL_A conversion on raw records, schema validation
- **Stats & Validation** - Cross-dataset stats, roundtrip validation, eval routing preview, MWCS preview
