# SympFlow Project Code

This repository contains the Python scripts, notebooks, for **“SympFlow”**.  


---

## Repository layout

| Path | Purpose |
|------|---------|
| **main.py** | Entry point for **unsupervised** training of SympFlows or MLPs.  This script allows to train a specified model, and saves the results in the folder *unsupervisedNetworks/*. The model, trainig, and dynamics specifics can be specified through an argparser.|
| **mainSupervised.py** | This script is analogous to the previous on but for **supervised** training where (noisy) training trajectories are available. Saves results in *supervisedNetworks/*. The model, trainig, and dynamics specifics can be specified through an argparser.|
| **generatePlots.py** | This script loads the last trained unsupervised model, and produces the requested plots. The plots to generate can be specified through the argparser. |
| **generatePlotsSupervised.py** | This script loads the last trained supervised model, and produces the requested plots. The plots to generate can be specified through the argparser. |
| **analyseHenonHeiles.ipynb** | Jupyter notebook create specifically to generate the Poincaré sections for the Hénon-Heiles system. |
| **scripts/** | Directory where all the scripts are collected|
| ├─ **networks.py** | Network architectures for SympFlow and MLP. |
| ├─ **training.py** | Training loop for the supervised and unsupervised experiments. |
| ├─ **sampling.py** | Draws initial conditions according to energy shells or uniform boxes, and creates the dataset for the supervised experiments. |
| ├─ **vector_fields.py** | Right‑hand sides \(\dot q,\dot p\) for a catalog of Hamiltonian systems (Simple Harmonic Oscillator, Damped Oscillator, and Hénon–Heiles) |
| ├─ **plotting.py** | Script that collects the plotting methods to generate the trajectories, energy behaviour, errors, and losses.|
| ├─ **utils.py** | Basic utilities for the experiments. |
| ├─ **experiments.py** | Script to specify the parameters in the dynamics. |

---

## Quick start

1. **Create and activate a virtual environment (recommended)**  
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate          # on Linux / macOS
   # or
   venv\Scripts\activate.bat       # on Windows
   ```

2. **Install Python dependencies**  
   ```bash
   pip install -e .
   pip install ".[torch]"
   ```

3. **Train a model** (unsupervised example)  
   ```bash
   python main.py \
          --ode_name FPUT \
          --m 1 \
          --omega 4 \
          --epochs 20001 \
          --save_path runs/fput_m1_w4/
   ```

   Replace the command‑line flags as needed; run `python main.py --help` for the full list.

4. **Generate plots for the trained model**  
   ```bash
   python generatePlots.py --plot_solutions --plot_loss --plot_errors --plot_energy
   ```
