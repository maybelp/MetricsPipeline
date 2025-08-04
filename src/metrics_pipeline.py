#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import unicodedata
import evaluate
import sacrebleu
import os
import argparse
from datetime import datetime

# --- Helper: Normalize Text ---
def normalize_text(s):
    return unicodedata.normalize('NFKC', s.strip())

# --- Metric Evaluation Function ---
def evaluate_system(preds, refs, system_name):
    results = []
    evaluate_references = [[ref] for ref in refs]
    sacrebleu_references = [refs]

    try:
        bleu_metric_eval = evaluate.load("bleu", tokenizer='13a')
        chrf_metric_eval = evaluate.load("chrf")
        ter_metric_eval = evaluate.load("ter", tokenizer='13a', normalize=False)

        eval_bleu = bleu_metric_eval.compute(predictions=preds, references=evaluate_references)['bleu']
        eval_chrf = chrf_metric_eval.compute(predictions=preds, references=evaluate_references)['score']
        eval_ter = ter_metric_eval.compute(predictions=preds, references=evaluate_references)['score']

        sacrebleu_bleu = sacrebleu.corpus_bleu(preds, sacrebleu_references, tokenize='13a').score
        sacrebleu_chrf = sacrebleu.corpus_chrf(preds, sacrebleu_references).score
        sacrebleu_ter = sacrebleu.corpus_ter(preds, sacrebleu_references).score

        results.append({
            "System": system_name, "Method": "evaluate.load()",
            "BLEU Score": eval_bleu, "chrF Score": eval_chrf, "TER Score": eval_ter
        })
        results.append({
            "System": system_name, "Method": "sacrebleu.corpus_*",
            "BLEU Score": sacrebleu_bleu, "chrF Score": sacrebleu_chrf, "TER Score": sacrebleu_ter
        })

        print(f" {system_name}:")
        print(f"  evaluate.load(): BLEU={eval_bleu:.2f}, chrF={eval_chrf:.2f}, TER={eval_ter:.2f}")
        print(f"  sacrebleu:       BLEU={sacrebleu_bleu:.2f}, chrF={sacrebleu_chrf:.2f}, TER={sacrebleu_ter:.2f}")
    except Exception as e:
        print(f"   Error evaluating {system_name}: {e}")
    return results

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate translation metrics.")
    parser.add_argument("--input", type=str, default="data/final_predictions_all.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default="results/metrics_summary.csv", help="Output CSV path")
    parser.add_argument("--json", type=str, default=None, help="Optional: Output path for JSON results")

    args = parser.parse_args()

    input_file = args.input
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_file = f"results/metrics_summary_{timestamp}.csv"

    if not os.path.exists(input_file):
        print(f" File not found: {input_file}")
        return

    column_map = {
        "mixtral_prediction": "Fine-Tuned Mixtral (Final)",
        "google_prediction": "Google Translate API",
        "nllb_prediction": "NLLB",
        "pretrained_prediction": "Base Mixtral (Pretrained)",
        "prediction_4000": "Mixtral Checkpoint-4000",
        "prediction_500": "Mixtral Checkpoint-500",
    }

    try:
        df = pd.read_csv(input_file)
        df.fillna("", inplace=True)
        df["reference"] = df["reference"].astype(str).apply(normalize_text)
    except Exception as e:
        print(f" Failed to load data: {e}")
        return

    refs = df["reference"].tolist()
    all_results = []

    for col, name in column_map.items():
        if col not in df.columns:
            print(f"⚠ Skipping missing column: {col}")
            continue
        preds = df[col].astype(str).apply(normalize_text).tolist()
        if len(preds) != len(refs):
            print(f"⚠ Skipping {name} due to length mismatch.")
            continue
        results = evaluate_system(preds, refs, name)
        all_results.extend(results)

    result_df = pd.DataFrame(all_results)

    # Save to file
    output_path = output_file
    try:
        result_df.to_csv(output_path, index=False)
        print(f"\n Results saved to: {output_path}")
        if args.json:
            try:
                result_df.to_json(args.json, orient="records", indent=2)
                print(f" JSON results also saved to: {args.json}")
            except Exception as e:
                print(f" Failed to save JSON: {e}")

    except Exception as e:
        print(f" Failed to save results: {e}")

    # Print to screen
    print("\n Final Results:")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
