import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import glob
from datetime import datetime

from scripts.networks import *
from scripts.vector_fields import *
from scripts.utils import *
import scripts.settings as settings
from scripts.experiments import *
from scripts.plotting import *

import sys

def get_last_trained_model(path):
    # Get the list of all files matching the pattern
    files = glob.glob(os.path.join(path, "trained_model_*.pt"))

    # Function to extract the timestamp from the filename
    def extract_timestamp(filename):
        basename = os.path.basename(filename)
        timestamp_str = basename[len("trained_model_"):-len(".pt")]
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

    # Find the latest file by comparing timestamps
    latest_file_name = max(files, key=extract_timestamp)
    print(f"\n\n {latest_file_name} \n\n")
    return latest_file_name

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plotting routines")
    parser.add_argument("--save_path", default="supervisedNetworks/")
    parser.add_argument("--ode_name", default="HenonHeiles") #among HenonHeiles,SimpleHO,DampedHO
    parser.add_argument("--final_time", default=100.0, type=float)
    parser.add_argument("--dt", default=1.0, type=float)
    parser.add_argument("--plot_solutions", action='store_true')
    parser.add_argument("--plot_energy", action='store_true')
    parser.add_argument("--plot_orbits", action="store_true")
    

    args = parser.parse_args()
    
    print("Plot solutions : ",args.plot_solutions)
    print("Plot long time energy behaviour : ",args.plot_energy)
    print("Plot orbits: ",args.plot_orbits)

    model_path = args.save_path + "savedModels"
    figure_path = args.save_path + 'figures/'
    
    if args.ode_name=="SimpleHO":
        system_parameters = SimpleHO_exp
    elif args.ode_name=="DampedHO":
        system_parameters = DampedHO_exp
    elif args.ode_name=="HenonHeiles":
        system_parameters = Henon_Heiles_exp
    else:
        warnings.warn("Dynamics not implemented.")
     
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
        nlayers=5,
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
                model_path = get_last_trained_model(path) #it appends the latest time stamp
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
                    figure_path=figure_path+"energy/"
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
                    figure_path=figure_path+"orbits/"
                )
        except:
            pass