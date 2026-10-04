#!/usr/bin/env bash
# One-click rebuild of every summary, table, figure, the paper PDF and the Word report from the result files.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
TECTONIC="${TECTONIC:-tectonic}"
mkdir -p ../logs
"$PY" select_eval.py loto > ../logs/final_select_loto.log 2>&1
"$PY" select_eval.py fcst > ../logs/final_select_fcst.log 2>&1
"$PY" analysis.py loto fcst > ../logs/final_analysis.log 2>&1
"$PY" ablation.py > ../logs/final_ablation.log 2>&1
"$PY" make_tables.py
"$PY" fig_data.py; "$PY" fig_repro.py; "$PY" fig_results.py
"$PY" -c "import fig_results as f; f.fig_fits(); f.fig_paths(); f.fig_insample(); f.fig_resid(); f.fig_transfer(); f.fig_sim(); f.fig_pooling(); f.fig_ablation(); f.fig_hybrid()"
"$PY" make_tables_zh.py
cd ../paper && "$TECTONIC" -X compile main.tex && "$TECTONIC" -X compile main_zh.tex && cd ../code
"$PY" make_report.py
echo "rebuilt: ../paper/main.pdf ../report.docx"
