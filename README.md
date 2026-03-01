# SympFlow Project Code

The contributors to this repository are: Davide Murari, Priscilla Canizares, Ferdia Sherry, and Zakhar Shumaylov.

This repository contains the Python scripts, notebooks, for “SympFlow”, a method introduced in the paper *Symplectic Neural Flows for Modeling and Discovery*.

```bibtex
@misc{canizares2024symplectic,
  title         = {Symplectic Neural Flows for Modeling and Discovery},
  author        = {Canizares, Priscilla and Murari, Davide and Sch{\"o}nlieb, Carola-Bibiane and Sherry, Ferdia and Shumaylov, Zakhar},
  year          = {2024},
  eprint        = {2412.16787},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2412.16787},
  url           = {https://arxiv.org/abs/2412.16787}
}
```

Codebase for the SympFlow experiments (training, evaluation, plots, and tables).

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
pip install ".[torch]"
```

Run all commands below from the repository root.

## Main Scripts

- `main.py`: unsupervised training (`pinnReg`, `pinnNoReg`, `hamReg`, `noHamReg`, `mixed`).
- `mainSupervised.py`: supervised training (`pinn`, `sympflow`).
- `generatePlots.py`: unsupervised solution/energy/error/loss plots.
- `generatePlotsSupervised.py`: supervised solution/energy/orbit plots.
- `scripts/evaluation/run_model_quality_eval.py`: quantitative sweep evaluation to CSV.
- `scripts/evaluation/run_model_quality_plots.py`: sweep plots from CSV.
- `scripts/evaluation/run_model_quality_table.py`: LaTeX-ready rows from one CSV.
- `scripts/evaluation/run_quality_comparison_table.py`: paired-comparison LaTeX rows (e.g., SRNN vs SympFlow).
- `scripts/evaluation/run_henon_heiles_poincare.py`: Hénon-Heiles Poincare sections (reference + networks) from shared ICs.

## Training Prerequisites (Before Plot/Eval Sweeps)

Most evaluation helpers require checkpoints already present in `supervisedNetworks/savedModels` and `unsupervisedNetworks/savedModels`.

### SimpleHO supervised (for Figure 5/Table SM2-style sweeps)

Train both `pinn` and `sympflow` for the combinations used by `iterativePlotsSupervised.sh`:
- epsilon sweep: `(N,M)=(100,50)`, `epsilon in {0.0,0.005,0.01,0.02,0.03,0.04,0.05}`
- N sweep: `(M,epsilon)=(50,0.0)`, `N in {10,50,100,150,200}`
- M sweep: `(N,epsilon)=(100,0.0)`, `M in {10,30,50,80}`

Example (single run):

```bash
python3 mainSupervised.py --ode_name SimpleHO --name_experiment sympflow --N 100 --M 50 --epsilon 0.0 --dt 1.0 --epochs 500 --number_layers 5
```

### SimpleHO unsupervised (for `iterativePlotsUnsupervised.sh`)

`iterativePlotsUnsupervised.sh` expects explicit files:
- `unsupervisedNetworks/savedModels/SimpleHO/<experiment>/nlayers_4.pt`
- `unsupervisedNetworks/savedModels/SimpleHO/<experiment>/nlayers_8.pt`
- `unsupervisedNetworks/savedModels/SimpleHO/<experiment>/nlayers_16.pt`

for each `<experiment>` in `hamReg,noHamReg,pinnReg,pinnNoReg`.

### Hénon-Heiles training

```bash
./iterativeTrainHenonHeiles.sh
```

### SRNN checkpoints

For `iterativePlotsSRNN.sh`, place SRNN checkpoints in `modelsSRNN/` with naming expected by `run_srnn_eval.py` (defaults in current setup are `N_1500_M_1_epsilon_<eps>.pt` for epsilon sweep).

## Shell Helpers

All top-level `.sh` scripts and what they do:

- `./iterativePlotsSupervised.sh [--ask|--reuse|--recompute]`
  - Builds supervised SimpleHO quality CSVs/plots/tables for `epsilon`, `N`, `M`.
  - `--reuse`: use existing CSVs if present.
  - `--recompute`: always regenerate CSVs.
  - `--ask` (default): prompt per sweep if CSV exists.

- `./iterativePlotsUnsupervised.sh [--ask|--reuse|--recompute]`
  - Builds unsupervised SimpleHO layer-sweep CSV/plots/table from `nlayers_{4,8,16}.pt`.
  - Same mode semantics as above.

- `./iterativePlotsSRNN.sh [--ask|--reuse|--recompute] [--only epsilon|N|M|epsilon,N,M]`
  - Runs SRNN eval + SympFlow eval, merges CSVs, then creates comparison plots/table.
  - `--only` restricts which sweep(s) to run.

- `./iterativePlotsHenonHeiles.sh`
  - Generates Hénon-Heiles unsupervised/supervised solution and energy plots, plus Poincare sections.

- `./iterativeTrainHenonHeiles.sh`
  - Trains the full Hénon-Heiles unsupervised and supervised experiment set used by the plotting helper above.

- `./testDampedOscillator.sh`
  - End-to-end damped-oscillator test script (train + plot) for selected damping values.

- `./run_unsupervised.sh`
  - Minimal unsupervised Hénon-Heiles training launcher (`hamReg`, `noHamReg`, `mixed`).

- `./comparison_srnn.sh`
  - Minimal SRNN-related supervised training loop over epsilon (legacy helper).

## One-Command Reproduction Helpers

- `./iterativePlotsSupervised.sh`: supervised SimpleHO quality sweeps (`epsilon`, `N`, `M`) + plots + table rows.
- `./iterativePlotsUnsupervised.sh`: unsupervised SimpleHO depth sweep (`N=layers`) + plots + table rows.
- `./iterativePlotsSRNN.sh`: SRNN vs SympFlow sweeps + merged plots + comparison table rows.
- `./iterativePlotsHenonHeiles.sh`: Hénon-Heiles solution/energy plots + Poincare sections.

## Paper Figure/Table Reproduction Map

Notes:
- Commands below assume required checkpoints already exist.
- Use the "Training Prerequisites" section above if checkpoints are missing.
- Some paper panels were assembled manually in LaTeX from multiple generated PDFs.

### Main Paper

| Paper item | How to generate with current code |
|---|---|
| Figure 3 (unsupervised SHO comparison) | `python3 generatePlots.py --save_path unsupervisedNetworks/ --ode_name SimpleHO --final_time 100 --dt 1.0 --plot_solutions --plot_errors --plot_energy` |
| Figure 4 (supervised SHO comparison) | `python3 generatePlotsSupervised.py --save_path supervisedNetworks/ --ode_name SimpleHO --final_time 100 --dt 1.0 --plot_solutions --plot_energy --plot_orbits` |
| Figure 5 (supervised SHO quality sweeps) | `./iterativePlotsSupervised.sh` |
| Figures 6-9 (damped HO supervised experiments) | Train with `mainSupervised.py` for `--ode_name DampedHO` and experiments `pinn/sympflow`, then `python3 generatePlotsSupervised.py --save_path supervisedNetworks/ --ode_name DampedHO --final_time 100 --dt 1.0 --plot_solutions --plot_energy` |
| Figure 10 (Hénon-Heiles unsupervised, incl. Poincare) | `./iterativePlotsHenonHeiles.sh` (unsupervised outputs under `unsupervisedNetworks/figures/`) |
| Figure 11 (Hénon-Heiles supervised, incl. Poincare) | `./iterativePlotsHenonHeiles.sh` (supervised outputs under `supervisedNetworks/figures/`) |

### Supplement

| Paper item | How to generate with current code |
|---|---|
| Figure SM1 (mixed-training SHO) | Train `main.py --ode_name SimpleHO --name_experiment mixed ...`, then `python3 generatePlots.py --save_path unsupervisedNetworks/ --ode_name SimpleHO --plot_solutions --plot_energy --plot_errors` |
| Figure SM2 (Hénon-Heiles unsupervised solutions) | `./iterativePlotsHenonHeiles.sh` |
| Figure SM3 (Hénon-Heiles supervised solutions) | `./iterativePlotsHenonHeiles.sh` |
| Figure SM4 (Hénon-Heiles reference Poincare section) | `python3 -m scripts.evaluation.run_henon_heiles_poincare --reference-only --output-dir unsupervisedNetworks/figures/poincareSections` |
| Figure SM5 (training/test losses) | `python3 generatePlots.py --save_path unsupervisedNetworks/ --ode_name SimpleHO --plot_loss` |
| Table SM1 (timings) | Read from generated timing logs: `supervisedNetworks/timingsTrainingPerEpoch.txt`, `unsupervisedNetworks/timingsTrainingPerEpoch.txt`, `unsupervisedNetworks/timingsInferecePer1s.txt` |
| Table SM2 (accuracy summary) | `./iterativePlotsSupervised.sh` + `./iterativePlotsUnsupervised.sh` then collect the produced `table_*.txt` rows in `supervisedNetworks/figures/modelQuality/supervised/` and `unsupervisedNetworks/figures/modelQuality/unsupervised/` |
| Table SM3 (SRNN vs SympFlow) | `./iterativePlotsSRNN.sh --only epsilon` (or `./iterativePlotsSRNN.sh --reuse --only epsilon`) |

## Quality Evaluation CLI Quick Examples

```bash
python3 -m scripts.evaluation.run_model_quality_eval --vary epsilon --is_supervised
python3 -m scripts.evaluation.run_model_quality_plots --csv supervisedNetworks/resultsVaryingEpsilon.csv --vary epsilon --is_supervised
python3 -m scripts.evaluation.run_model_quality_table --csv supervisedNetworks/resultsVaryingEpsilon.csv --vary epsilon --output-txt supervisedNetworks/figures/modelQuality/supervised/table_varyingEpsilon.txt
```
