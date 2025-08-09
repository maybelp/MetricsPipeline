# MetricsPipeline

## Translation Quality Evaluation
Evaluate machine translation outputs with BLEU, chrF, and TER using both 🤗 evaluate and sacrebleu.
Designed for comparing multiple systems (columns) against a single reference column in one CSV.

## Highlights
Metrics: BLEU, chrF, TER (via both evaluate and sacrebleu)

Text normalization with Unicode NFKC

Simple CLI (python -m src.run_all …)

PyTest suite (tests/)

Works on Hydra or locally; supports data symlinks

## Project Structure
```
MetricsPipeline/
├── data/                         # Put your CSVs here (or symlink them)
│   ├── final_predictions_all.csv
│   └── final_predictions_all_FLORES.csv
├── results/                      # Outputs are written here
│   ├── metrics_summary.csv
│   ├── flores_summary.csv
│   └── flores_summary.json
├── src/
│   ├── __init__.py
│   ├── metrics_pipeline.py       # Core logic (normalize, evaluate, compare)
│   └── run_all.py                # CLI entry point
├── tests/
│   ├── test_basic.py
│   └── test_normalization.py
├── README.md
└── requirements.txt


```
## Installation
### 1) (Recommended) Create/activate a virtual environment
### 2) Install dependencies
pip install -r requirements.txt
If you’re on Hydra and your CSV lives elsewhere (e.g.,~/mixtral_data/…) , create a symlink so the pipeline can see it: 
### ln -s ~/mixtral_data/final_predictions_all.csv data/final_predictions_all.csv
### ln -s ~/mixtral_data/final_predictions_all_FLORES.csv data/final_predictions_all_FLORES.csv

## Input Format
The pipeline expects one CSV with:

A reference column named: reference

One or more prediction columns using any of these names (already mapped in code):

mixtral_prediction → “Fine-Tuned Mixtral (Final)”

pretrained_prediction → “Base Mixtral (Pretrained)”

google_prediction → “Google Translate API”

nllb_prediction → “NLLB”

prediction_4000 → “Mixtral Checkpoint-4000”

prediction_500 → “Mixtral Checkpoint-500”

Missing columns are safely skipped with a warning.
### Quick Start
Run with defaults
### python -m src.run_all

This looks for data/final_predictions_all.csv and writes results/metrics_summary.csv
## What You'll Get
### A CSV like:
System,Method,BLEU Score,chrF Score,TER Score
Fine-Tuned Mixtral (Final),evaluate.load(),0.1459,33.6749,91.8785
Fine-Tuned Mixtral (Final),sacrebleu.corpus_*,14.5855,33.6749,91.8785
Google Translate API,evaluate.load(),0.1522,36.5972,77.6355
Google Translate API,sacrebleu.corpus_*,15.2157,36.5972,77.6355
...
### Notes:
evaluate.load() BLEU is scaled 0–1; sacrebleu BLEU is 0–100. This is expected.

You may see sacrebleu warnings about detokenization; they’re informational.

### Testing
From the project root: 
# Ensure the package can be imported
PYTHONPATH=. pytest tests/
you should see all tests passing: 
================================== 6 passed ==================================
### Common Issues & Fixes
ModuleNotFoundError: No module named 'src'

Run with module mode: python -m src.run_all

Or set PYTHONPATH=. before running tests.

“File not found: data/<...>.csv”

Place your CSV in data/ or create a symlink (see Installation).

SacreBLEU tokenization warning

Informational; you can ignore for quick comparisons.

For strict scoring, ensure your inputs are properly detokenized.

### Reproducibility Tips
Keep your CSVs under version control (if small) or track them via DVC / symlinks.

Pin versions in requirements.txt (already included).

Save the console output and results/*.csv alongside each run.
### License
Add your chosen license (MIT is common for research code).
Create a LICENSE file and link it here.
## Acknowledgements
Hugging Face evaluate

sacrebleu

Your university’s Hydra cluster administrators 💙
