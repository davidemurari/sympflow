import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import glob
import re
from datetime import datetime

from scripts.networks import *
from scripts.vector_fields import *
from scripts.utils import *
import scripts.settings as settings
from scripts.experiments import *
from scripts.plotting import *

import sys

def _parse_checkpoint_name(filename):
    """Parse a supervised checkpoint filename into metadata.

    Inputs:
        filename (str): Checkpoint filename or full path.

    Returns:
        dict | None: Parsed metadata dictionary when the naming format is
        supported, otherwise ``None``.
    """
    basename = os.path.basename(filename)

    # New supervised naming:
    # trained_model_eps_<eps>_N_<N>_M_<M>_<timestamp>.pt
    # or trained_model_eps_<eps>_N_<N>_M_<M>.pt
    m = re.match(
        r"^trained_model_eps_(?P<epsilon>[-+0-9.eE]+)_N_(?P<N>\d+)_M_(?P<M>\d+)(?:_(?P<ts>\d{8}_\d{6}))?\.pt$",
        basename,
    )
    if m:
        ts = m.group("ts")
        return dict(
            kind="supervised",
            epsilon=float(m.group("epsilon")),
            N=int(m.group("N")),
            M=int(m.group("M")),
            timestamp=(datetime.strptime(ts, "%Y%m%d_%H%M%S") if ts else None),
        )

    # Legacy naming:
    # trained_model_<timestamp>.pt
    m = re.match(r"^trained_model_(?P<ts>\d{8}_\d{6})\.pt$", basename)
    if m:
        return dict(
            kind="legacy",
            epsilon=None,
            N=None,
            M=None,
            timestamp=datetime.strptime(m.group("ts"), "%Y%m%d_%H%M%S"),
        )

    return None


def resolve_model_checkpoint(path, n=None, m=None, epsilon=None):
    """Resolve a checkpoint path with optional supervised-config filtering.

    Inputs:
        path (str): Directory containing checkpoint files.
        n (int | None): Optional dataset-size filter ``N``.
        m (int | None): Optional per-trajectory-samples filter ``M``.
        epsilon (float | None): Optional noise-level filter.

    Returns:
        str: Path to the selected checkpoint file.
    """
    files = glob.glob(os.path.join(path, "trained_model*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint matching 'trained_model*.pt' in {path}")

    parsed = [(f, _parse_checkpoint_name(f)) for f in files]
    parsed = [(f, meta) for f, meta in parsed if meta is not None]
    if not parsed:
        raise FileNotFoundError(f"Found checkpoints in {path}, but none with a supported naming convention.")

    use_filter = any(v is not None for v in (n, m, epsilon))
    if use_filter:
        filtered = []
        for f, meta in parsed:
            # Ignore legacy names when explicit supervised config is requested.
            if meta["kind"] != "supervised":
                continue
            if n is not None and meta["N"] != int(n):
                continue
            if m is not None and meta["M"] != int(m):
                continue
            if epsilon is not None and not np.isclose(meta["epsilon"], float(epsilon), atol=1e-12, rtol=0.0):
                continue
            filtered.append((f, meta))

        timestamped = [(f, meta["timestamp"]) for f, meta in filtered if meta["timestamp"] is not None]
        if timestamped:
            return max(timestamped, key=lambda x: x[1])[0]

        if filtered:
            return max((f for f, _ in filtered), key=os.path.getmtime)

        raise FileNotFoundError(
            f"No checkpoint found in {path} matching requested configuration "
            f"(N={n}, M={m}, epsilon={epsilon})."
        )

    # No config filter: keep current behavior (latest timestamp if available).
    timestamped = [(f, meta["timestamp"]) for f, meta in parsed if meta["timestamp"] is not None]
    if timestamped:
        return max(timestamped, key=lambda x: x[1])[0]
    return max((f for f, _ in parsed), key=os.path.getmtime)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plotting routines")
    parser.add_argument("--save_path", default="supervisedNetworks/")
    parser.add_argument("--ode_name", default="HenonHeiles") #among HenonHeiles,SimpleHO,DampedHO
    parser.add_argument("--final_time", default=100.0, type=float)
    parser.add_argument("--dt", default=1.0, type=float)
    parser.add_argument("--ll", default=None, type=float, help="Optional damping coefficient lambda for DampedHO.")
    parser.add_argument("--number_layers", default=5, type=int, help="Number of layers of the model architecture to load.")
    parser.add_argument("--N", default=None, type=int, help="Optional supervised dataset size filter for checkpoint selection.")
    parser.add_argument("--M", default=None, type=int, help="Optional supervised time-samples filter for checkpoint selection.")
    parser.add_argument("--epsilon", default=None, type=float, help="Optional supervised noise-level filter for checkpoint selection.")
    parser.add_argument("--plot_solutions", action='store_true')
    parser.add_argument("--plot_energy", action='store_true')
    parser.add_argument("--plot_orbits", action="store_true")
    

    args = parser.parse_args()
    
    print("Plot solutions : ",args.plot_solutions)
    print("Plot long time energy behaviour : ",args.plot_energy)
    print("Plot orbits: ",args.plot_orbits)

    model_path = args.save_path + "savedModels"
    figure_path = args.save_path + 'figures/'
    
    try:
        system_parameters = get_system_parameters(args.ode_name, ll=args.ll)
    except ValueError:
        warnings.warn("Dynamics not implemented.")
        raise
     
    vector_field_class = globals()[system_parameters['vec_field_name']]
    vec = vector_field_class(system_parameters)

    settings.paths = dict(
        model = model_path,
        figure = figure_path
        )
 
    print(settings.paths['model'])
    
    torch.manual_seed(1)
    np.random.seed(1)
    dtype=torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Make sure that these parameters are the same as the ones used during training
    model_parameters = dict(
        hidden_nodes=10,
        act_name="tanh",
        nlayers=args.number_layers,
        device=device,
        dtype=dtype,
        d=vec.ndim_total,
    )

    training_parameters = dict(
        dt=args.dt,
        tf=args.final_time,
        t0=-0.1,
        n_train=700,
        n_test=5000,
        epochs=10001,
        device=device,
        dtype=dtype,
        lr = 5e-3
    )

    if not os.path.exists("supervisedNetworks/figures"):
        os.mkdir("supervisedNetworks/figures")
    if not os.path.exists("supervisedNetworks/figures/energy"):
        os.mkdir("supervisedNetworks/figures/energy")
    if not os.path.exists("supervisedNetworks/figures/solutions"):
        os.mkdir("supervisedNetworks/figures/solutions")
    if not os.path.exists("supervisedNetworks/figures/orbits"):
        os.mkdir("supervisedNetworks/figures/orbits")
   
   
    ode_name = args.ode_name 
    
    if ode_name=="DampedHO":
        ode_name=ode_name+f"_{str(vec.ll)}"
    
    for name_experiment in ["pinn","sympflow"]:
        
        try:
            if args.plot_solutions or args.plot_energy or args.plot_orbits:
                
                path = f"{settings.paths['model']}/{ode_name}/{name_experiment}"
                model_path = resolve_model_checkpoint(path, n=args.N, m=args.M, epsilon=args.epsilon)
                print(f"Selected checkpoint: {model_path}")
                print(path)
                if name_experiment=="pinn":
                    model = genericNet(model_parameters,vec=vec, dt=training_parameters['dt']) #Usual network
                else:
                    model = sympNet(model_parameters,vec=vec, dt=training_parameters['dt']) #Symplectic network
                model.to(device);

                model.load_state_dict(torch.load(model_path,map_location=device),strict=True) #Strict false because there is h_0 which was not there before
                print("Model loaded correctly")
                print("Generating the solutions")
                q0,pi0,tf,dtype,device = system_parameters['q0'], system_parameters['pi0'], training_parameters['tf'], training_parameters['dtype'], training_parameters['device']
                                
                vec,t_eval,sol_scipy,sol_network = generate_solutions(vec,q0,pi0,tf,model,dtype,device)

            if args.plot_energy:
                print("Plotting long time energy behaviour")
                plotLongTimeEnergy(
                    vec,
                    ode_name,
                    name_experiment,
                    t_eval,
                    sol_scipy,
                    sol_network,
                    is_supervised=True,
                    figure_path=figure_path
                )

            if args.plot_solutions:
                print("Plotting the solutions")
                plotSolutions(
                    vec,
                    ode_name,
                    name_experiment,
                    t_eval,
                    sol_scipy,
                    sol_network,
                    is_supervised=True,
                    figure_path=figure_path
                )

            if ode_name=="SimpleHO" or ode_name==ode_name and args.plot_orbits:
                plotSolutions_2d(
                    vec,
                    ode_name,
                    name_experiment,
                    t_eval,
                    sol_scipy,
                    sol_network,
                    is_supervised=True,
                    figure_path=figure_path
                )
        except Exception as exc:
            print(f"Skipping experiment '{name_experiment}' due to: {exc}")
