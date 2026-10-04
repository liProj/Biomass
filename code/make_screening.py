"""Build Biomass/screening.csv: per-paper assessment of the 80 most recent Biomass papers.
Scores: s_data/s_time/s_ease in 0-5, s_figs = figure count capped at 10.
score = 0.4*s_data + 0.3*s_time + 0.2*s_ease + 0.1*s_figs   (same weighting as Seeds/Analytica/Birds)
Data categories (from the Data Availability Statement, checked against the files actually downloaded):
  REV  = review / no new data;  REQ = data only on request (not public);  ART = 'in the article' (summary
  tables only, raw data not released);  SUP = supplementary file actually downloaded and inspected."""
import pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__)); JD = os.path.dirname(HERE)

REV = ("none (review / no new data)", "-", "-", "literature synthesis", "n/a", "-", "-", "n/a (no dataset)", 0, 1, 1)
def req(method, idea="ML surrogate of the experimental response", why="raw data on request only"):
    return ("NOT public (on request)", "-", "unknown", method, f"blocked: {why}", idea, "low-medium", "n/a (data not public)", 1, 2, 2)
def art(method, size, idea="Bayesian/GP response-surface model", note="only summary tables in the article"):
    return ("in article only (summary tables / figures)", "article tables", size, method,
            f"partial: {note}", idea, "low-medium", "weak (tiny N, raw replicates missing)", 2, 3, 3)

A = {
82: req("ABE co-culture fermentation batch experiments; yields/productivities", "kinetic hybrid model of co-culture"),
79: req("gasification equilibrium + HCCI engine simulation + techno-economics", "surrogate of the HCCI simulator"),
78: ("derived from cited sources / IBGE statistics", "public national statistics", "small", "review-type inventory of residues", "n/a", "-", "low", "n/a", 1, 2, 2),
77: REV, 76: REV, 72: REV, 71: REV, 67: REV, 64: REV, 63: REV, 57: REV, 42: REV, 34: REV, 33: REV, 32: REV,
24: REV, 18: REV, 13: REV, 10: REV, 11: REV, 9: REV, 1: REV,
62: ("none (scoping review of cookstove CFD; data on request)", "-", "-", "scoping review", "n/a", "-", "-", "n/a", 0, 1, 1),
15: ("none stated ('no new data' DAS although experimental)", "-", "-", "arabinoxylan film mechanical tests", "blocked", "-", "low", "n/a", 1, 2, 2),
75: art("TGA isoconversional kinetics (FWO/KAS) + thermodynamics", "TGA curves shown as figures only",
        "neural/physics kinetic deconvolution", "raw TGA curves not released"),
74: req("photobioreactor CO2 uptake; spectral optimization", "growth-kinetic hybrid model"),
73: req("in-line NIR spectroscopy + PLS for woodchip moisture", "1D-CNN / spectral foundation model", "NIR spectra on request"),
69: req("proximate/ultimate/TGA/FTIR characterization of plantain peel", "-"),
70: art("TG-FTIR/TG-MS pyrolysis kinetics of hydrochar at five heating rates", "TG curves in figures",
        "physics-informed kinetic network", "raw TG/MS signals not released"),
68: req("pellet pyrolysis product distribution", "-"),
66: ("Supplementary (Taguchi L9 design results)", "supp s001 (if present)", "9 runs x responses", "Taguchi L9 screening + ANOVA",
     "easy but N=9", "GP / Bayesian design analysis", "low", "weak (9 runs)", 3, 4, 4),
65: art("dual-chamber MFC performance comparison", "polarization curves", "-", "time series only as figures"),
61: ("Supplementary (spreadsheets + robustness script)", "supp s001", "small decision matrix",
     "energy/exergy/emergy indices + MCDA (Monte Carlo, Pareto, critical weights)", "easy",
     "probabilistic MCDA", "low", "no predictive metric to beat", 4, 4, 3),
60: ("Supplementary (34-case sub-index matrix + calculations)", "supp s001", "34 plants x sub-indices",
     "Biopolygeneration Diagnostic Index gap decomposition", "easy",
     "learned weighting / latent-factor index", "low-medium", "no predictive metric to beat", 4, 4, 3),
59: req("prototype stove tests", "-"),
58: ("partly public: Table S1 has LBO data only; SR/AP/HBO data are in an earlier paper [7]", "supp s001 + ref [7]",
     "27 HTL runs", "PCA + linear regression (MATLAB); D_mod partition coefficient",
     "medium: needs merging with prior-paper data; in-sample metrics only",
     "multi-output GP / TabICL with physics constraints", "medium", "plausible but data split across two papers", 3, 3, 3),
56: req("3x3 factorial pyrolysis of rice husk; ANOVA", "GP response surface"),
55: art("Box-Behnken lignin extraction + bioethanol", "BBD runs", "GP/Bayesian optimisation", "replicates not released"),
54: ("Supplementary PDF (Table S1: 75 strains screening); fermentation data on request", "supp s001 pdf", "75 strains x 4 activities",
     "strain screening + SSF optimisation", "partial", "-", "low", "no model to beat", 2, 3, 3),
53: req("1D anisotropic particle pyrolysis simulation (CRECK kinetics)", "neural-operator surrogate of the simulator", "simulation outputs on request"),
51: req("DSSC fabrication with Moringa chlorophyll", "-"),
52: art("hydrochar surface chemistry and metal binding isotherms", "isotherm points in figures"),
50: ("public (Supplementary File S1 xlsx: raw long-format data 480 rows + all statistical outputs; S2 Python script)",
     "supp s001: Supplementary_File_S1_Dataset_and_statistical_outputs.xlsx + S2 script", "480 rows (5 traits x 6 trt x 4 yr x 4 blocks)",
     "repeated-measures MixedLM (statsmodels) + Tukey HSD + economics",
     "easy (data + script public)", "Bayesian hierarchical model with AR(1) year effects (fixes the non-converged MixedLM)",
     "low-medium", "no predictive performance metric reported (inferential study); improvement not measurable as 'beating' a score", 5, 5, 4),
49: ("NOT reconstructable: Table A1 lists 20 feedstocks with T and yield RANGES only; 72 points not released; code on request",
     "Table A1 (ranges)", "72 points (not recoverable)", "ANN vs physics-informed NN vs rough-set ML for char yield",
     "blocked: per-point (T, yield) pairs missing -> fails exact data check", "monotone GP / TabPFN-style prior",
     "medium", "n/a (data check fails: 72 rows cannot be reproduced)", 1, 3, 3),
48: req("transesterified ester insulating fluids properties", "-"),
47: art("BMP tests + derivative-based kinetic landmarks (Gompertz etc.)", "cumulative methane curves in figures",
        "hierarchical kinetic model", "BMP curves not tabulated"),
45: ("Supplementary (DFT geometries only); adsorption data in appendix tables", "supp s001 pdf", "isotherm/kinetic tables",
     "adsorption isotherms + kinetics + DFT", "partial", "-", "low", "weak", 2, 3, 3),
46: req("boiler energy/exergy/dynamic/emergy framework", "-"),
44: ("NOT public (institutional/privacy restrictions)", "-", "survey", "SEM / survey of farmer adoption", "blocked", "-", "-", "n/a", 0, 2, 2),
43: art("lozenge formulation + sensory", "few formulations"),
41: ("NOT public (confidential industrial partner)", "-", "time series", "multi-step forecasting of Spirulina biomass (ML)",
     "blocked: confidential", "temporal foundation model (Chronos/TimesFM)", "medium", "n/a (data not public)", 0, 2, 3),
40: art("IAA production by Streptomyces, docking", "few conditions"),
39: art("polyol liquefaction of bay laurel", "few runs"),
38: ("none ('no data was used')", "-", "-", "liquefaction experiments + empirical model", "blocked", "-", "-", "n/a", 0, 2, 2),
37: ("Supplementary Table S1 (complete physicochemical dataset)", "supp s001 pdf", "~20 samples x 20 properties",
     "descriptive characterization", "easy", "-", "low", "no model to beat", 3, 4, 4),
36: art("GHG flux from fish waste composting", "flux time series in figures"),
35: ("Supplementary (RSM figures/tables only)", "supp s001 pdf", "RSM runs", "severity factor + RSM optimisation",
     "partial", "GP response surface", "low", "weak (tiny N)", 2, 3, 3),
31: ("Supplementary (LCI tables only)", "supp s001 pdf", "LCI", "factorial briquette tests + LCA", "partial", "-", "low", "weak", 2, 3, 3),
30: art("nanobiochar phenol adsorption isotherms", "isotherms"),
29: art("magnetic immobilization CBP", "few runs"),
28: req("lignin-modified asphalt binder tests", "-"),
27: art("TPAD co-digestion", "few reactors"),
26: req("Chlorella cultivation on aquaponics effluent", "-"),
25: ("Supplementary (agricultural statistics, survey responses, normalized matrix, scenarios)", "supp s001 (6 files)", "small matrices",
     "spatial MCDA (TOPSIS-like) for substrate selection", "easy", "-", "low", "no predictive metric", 4, 4, 3),
23: req("1D transient grate-boiler CFD-like model", "neural surrogate"),
22: ("Supplementary Table S1 (13 RSM runs)", "supp s001 pdf", "13 runs x 3 responses", "central composite RSM (quadratic)",
     "easy", "GP / Bayesian RSM with LOO", "low", "possible but N=13", 4, 5, 4),
21: ("Supplementary PDF (summary statistics only)", "supp s001 pdf", "summary stats", "MSR + biomethanation BES", "partial", "-", "low", "weak", 2, 3, 3),
20: ("Supplementary (ANOVA table only); raw data 'available' per DAS but not in file", "supp s001 pdf (1 page)", "2^2 design",
     "full factorial 2^2 + ANOVA + kinetic/thermodynamic denaturation", "partial", "-", "low", "weak", 2, 3, 3),
19: ("NOT public (privacy / company ownership)", "-", "-", "cost-benefit analysis", "blocked", "-", "-", "n/a", 0, 2, 2),
17: ("Supplementary CSVs (NREL TMY solar, WIND Toolkit, load SI 3)", "supp s001 (3 CSV, 3 MB)", "8760-h series",
     "HOMER-type hybrid PV-wind-biomass system sizing", "medium (commercial optimiser)",
     "MILP/stochastic optimisation", "medium", "possible but not a predictive benchmark", 4, 3, 2),
16: req("bubbling fluidized-bed gasification runs", "-"),
14: req("peach-stone carbon production", "-"),
12: ("public: Supplementary PDF Tables S1-S7 give every reaction profile (t, c, Z, M, D, T) used in the paper (digitised from 4 literature sources)",
     "supp s001: biomass-4043327-supplementary.pdf (28 pages)", "8 batch reaction profiles, 7-25 time points each (~500 numbers)",
     "two-step semi-mechanistic Ping Pong bi bi kinetics: least-squares selectivity fingerprinting (SM, SD) then dynamic ODE fit (kA, FA, FAZ, FAM, FAD); Scilab; Fobj/BIC + Hessian CIs",
     "easy: ODE system fully specified (Eqs 16-19, 30-34), every fitted value tabulated (Tables 1-6) for exact verification",
     "joint (single-step) hierarchical hybrid kinetics: mechanistic Ping Pong core + monotone neural/learned enzyme-deactivation & inhibition modifiers, shared selectivities across conditions (meta-learned prior), Arrhenius/aw coupling, Bayesian uncertainty, held-out prediction",
     "medium-high", "high: original reports in-sample Fobj/BIC per dataset only and treats datasets separately; joint/hybrid model can lower Fobj and BIC and be tested out-of-sample", 5, 5, 4),
8: art("dairy-effluent AD at two temperatures", "few reactors"),
7: req("rotary-kiln pyrolysis of sewage sludge with CaO", "-"),
4: ("processed spreadsheets on request; sources public", "public statistics", "inventory", "techno-economic biogas inventory", "medium", "-", "low", "n/a", 2, 2, 2),
5: ("Supplementary PDF (Py-GC/MS compound identification tables)", "supp s001 pdf", "compound tables", "catalytic fast pyrolysis Py-GC/MS",
    "partial", "-", "low", "no model to beat", 3, 3, 3),
6: ("on request (supp has characterization only)", "supp s001 pdf", "isotherms", "hydrochar dye adsorption isotherms/kinetics", "partial", "-", "low", "weak", 2, 3, 3),
3: art("activated carbon synthesis for biodiesel dry wash", "few runs"),
2: ("Supplementary (Py-GC/MS compound lists)", "supp s001 pdf", "compound tables", "TGA + Py-GC/MS of greenhouse waste", "partial", "-", "low", "no model to beat", 3, 3, 3),
}

raw = pd.read_csv(f"{JD}/screen_raw.csv")
meta = pd.read_csv(f"{JD}/latest50_crossref.csv")
rows = []
for _, r in raw.iterrows():
    a = int(r.article)
    v = A.get(a)
    if v is None:
        raise SystemExit(f"no assessment for #{a}")
    s_data, s_time, s_ease = v[8], v[9], v[10]
    s_figs = min(int(r.n_fig), 10)
    score = round(0.4*s_data + 0.3*s_time + 0.2*s_ease + 0.1*s_figs, 2)
    rows.append(dict(article=a, doi=r.doi, published=r.published, title=r.title,
                     type="review" if v is REV or "review" in str(r.title).lower() else "article",
                     data_availability=v[0], data_location=v[1], data_size=v[2], original_method=v[3],
                     reproduction_difficulty=v[4], new_method_idea=v[5], novelty=v[6],
                     beat_original_feasibility=v[7], n_figures=int(r.n_fig), n_tables=int(r.n_tab),
                     s_data=s_data, s_time=s_time, s_ease=s_ease, s_figs=s_figs, score=score))
df = pd.DataFrame(rows).sort_values(["score", "s_data"], ascending=False)
df.insert(0, "rank", range(1, len(df) + 1))
df.to_csv(f"{JD}/screening.csv", index=False)
print(len(df), "papers;", df[["rank", "article", "score", "s_data", "n_figures", "title"]].head(12).to_string(index=False, max_colwidth=60))
