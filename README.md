# Biomass — PPK-FIP

Research code and accompanying materials for PPK-FIP, a kinetic modelling framework for lipase-catalysed sequential esterification. The project compares an initial two-step method with joint maximum a posteriori estimation using fingerprint-informed priors, pseudo-processivity and enzyme deactivation.

本仓库包含论文的模型实现、拟合与验证代码、整理后的实验数据、已计算结果、绘图脚本以及中英文论文文件。

## Repository contents

| Path | Contents |
| --- | --- |
| `code/` | All 37 original analysis, modelling, plotting and reporting scripts |
| `data/profiles.csv` | 198 rows parsed from the cited supplementary material, including profiles excluded from the main analysis |
| `results/` | Saved fits, cross-validation predictions, diagnostics, ablations and simulation results |
| `figures/` | Paper figures and alternative workflow illustrations |
| `paper/` | Current English and Chinese LaTeX sources, tables, bibliography and PDFs |
| `data_check.md` | Audit of reported quantities against the public data |
| `screening.csv`, `screen_raw.csv`, `latest50_crossref.*` | Literature-screening records and metadata |
| `report.docx` | Chinese experiment report |

Downloaded literature PDFs, extracted reference texts, downloaded supplementary archives, old manuscript backups, runtime caches and logs are not tracked. They remain in the local working directory.

## Installation

Use a Python environment compatible with NumPy, SciPy and Numba. Python 3.11 or 3.12 is a reasonable starting point; the original environment was not locked and exact package versions are not recorded.

~~~bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

For GP, MLP, TabICL and neural ODE baselines, also install:

~~~bash
python -m pip install -r requirements-ml.txt
~~~

TabICL may download pretrained weights on first use. PDF compilation additionally requires Tectonic on PATH, its LaTeX resources and the fonts requested by the manuscript. The shell rebuild script requires Bash (for example, Linux, WSL or Git Bash).

## Quick start

Run commands from the repository root:

~~~bash
python code/verify_data.py
~~~

To rebuild summaries, tables, figures, manuscript PDFs and the Word report from the saved results:

~~~bash
bash code/finalize.sh
~~~

The rebuild updates generated outputs. It does not rerun all expensive fitting and cross-validation experiments. It uses `python` and `tectonic` from PATH; set `PYTHON` and `TECTONIC` to override their executable paths.

## Experiment entry points

| Task | Script |
| --- | --- |
| Initial fingerprint and dynamic fits | `code/repro.py`, `code/repro_compare.py` |
| Identifiability and solver diagnostics | `code/diagnostics.py` |
| Joint fits and final parameters | `code/insample.py`, `code/final_fit.py`, `code/ci.py` |
| Cross-validation and forward extrapolation | `code/cv.py`, `code/cv_all.py`, `code/cv_map.py` |
| Component ablation, pooling and hybrid models | `code/cv_ablate.py`, `code/cv_pooled.py`, `code/cv_hybrid.py` |
| Data-driven baselines | `code/blackbox.py` |
| Cross-condition transfer and parameter recovery | `code/transfer.py`, `code/sim.py` |
| Model selection and result summaries | `code/select_eval.py`, `code/analysis.py`, `code/ablation.py` |

Scripts with a protocol argument use `loto` for leave-one-time-point-out validation and `fcst` for forward extrapolation. Example commands (these recompute and overwrite corresponding result files):

~~~bash
python code/cv_all.py loto
python code/cv_all.py fcst
python code/blackbox.py loto GP,MLP-ens,TabICL,NODE 4
python code/transfer.py
python code/sim.py 60
~~~

Many fitting scripts use multiprocessing. The scripts that read `NPROC` accept it as an environment variable; some other scripts specify their worker count directly. See each script for its arguments and prerequisites. The saved `results/` files allow inspection without rerunning the full experiment suite.

## Data provenance

The experimental profiles are derived from the supplementary tables of Nishimura et al., *A Semi-Mechanistic Approach to Modeling Lipase-Catalyzed Processes with Multiple Competing Reactions*, **Biomass** 2026, 6, 12. [DOI: 10.3390/biomass6010012](https://doi.org/10.3390/biomass6010012). The source study in turn attributes the experimental datasets to the studies identified in `data/profiles.csv` and the manuscript bibliography.

The parsed CSV is included. To re-extract it, download the supplementary PDF from the source article and place it at:

~~~text
supp/biomass-06-00012-s001/biomass-4043327-supplementary.pdf
~~~

Then run `python code/parse_supp.py`. The literature-screening utility uses locally downloaded reference texts in `texts/`; those downloads are not part of this repository.

## Publication check

Before the initial GitHub upload, all 36 Python files were checked for syntax, and the baseline model was checked for data loading, finite predictions and conservation of total TMP species. The full experiment suite was not rerun for this upload. The dependency lists are inferred from source imports, not an exact environment lock.
