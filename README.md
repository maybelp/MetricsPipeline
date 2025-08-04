# MetricsPipeline
#  MetricsPipeline

A modular and testable Python pipeline to evaluate translation systems using BLEU, chrF, and TER metrics via both `evaluate` and `sacrebleu`. Designed for research and experimentation on FLORES datasets, with support for checkpoint comparisons and automated reporting.

---

##  Features

- Compare multiple translation systems (e.g. fine-tuned, NLLB, Google Translate)
- Uses both `evaluate` and `sacrebleu` libraries
- CLI support for flexible input/output
- Detokenization-aware normalization
- Automated test coverage with `pytest`

---

##  Project Structure
''' 
MetricsPipeline/
├── data/ # Input prediction CSVs
│ ├── final_predictions_all.csv
│ └── final_predictions_all_FLORES.csv
│
├── results/ # Evaluation outputs
│ ├── metrics_summary.csv
│ ├── flores_summary.csv
│ └── flores_summary.json
│
├── src/
│ ├── utils/
│ │ └── init.py
│ ├── metrics_pipeline.py # Core metric logic
│ └── run_all.py # CLI wrapper to run the pipeline
│
├── tests/ # Unit tests using pytest
│ ├── test_basic.py
│ └── test_normalization.py
│
├── .gitignore
├── .gitlab-ci.yml # Optional GitLab CI configuration
├── requirements.txt
└── README.md #
'''
##  Installation

# bash
pip install -r requirements.txt

# Run evaluation pipeline:
'''  
python -m src.run_all --input data/final_predictions_all.csv --output results/metrics_summary.csv

'''
# Evaluation Metrics
'''
Metric	Description	Tools
BLEU	Precision-based n-gram overlap	evaluate, sacrebleu
chrF	Character n-gram F-score	evaluate, sacrebleu
TER	Edit distance-based translation metric
'''
# Run unit tests with:
'''
PYTHONPATH=. pytest tests/
'''
