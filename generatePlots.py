import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import glob
import sys

from datetime import datetime
from scripts.networks import *
from scripts.vector_fields import *
from scripts.utils import *
import scripts.settings as settings
from scripts.experiments import *
from scripts.plotting import *


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
    return latest_file_name
    
def get_last_losses(path):
    # Get the list of all files matching the pattern
    files = glob.glob(os.path.join(path, "TrainingLosses_*.txt"))

    # Function to extract the timestamp from the filename
    def extract_timestamp(filename):
        basename = os.path.basename(filename)
        timestamp_str = basename[len("TrainingLosses_"):-len(".txt")]
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

    # Find the latest file by comparing timestamps
    latest_TrainingLosses = max(files, key=extract_timestamp)
    
    files = glob.glob(os.path.join(path, "TestLosses_*.txt"))

    # Function to extract the timestamp from the filename
    def extract_timestamp(filename):
        basename = os.path.basename(filename)
        timestamp_str = basename[len("TestLosses_"):-len(".txt")]
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

    # Find the latest file by comparing timestamps
    latest_TestLosses = max(files, key=extract_timestamp)
    
    return latest_TrainingLosses, latest_TestLosses

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plotting routines")
    parser.add_argument("--save_path", default="unsupervisedNetworks/")
    parser.add_argument("--ode_name", default="HenonHeiles") #among HenonHeiles,SimpleHO,DampedHO
    parser.add_argument("--final_time", default=100.0, type=float)
    parser.add_argument("--dt", default=1.0, type=float)
    parser.add_argument("--plot_loss", action='store_true')
    parser.add_argument("--plot_errors", action='store_true')
    parser.add_argument("--plot_solutions", action='store_true')
    parser.add_argument("--plot_energy", action='store_true')
    

    args = parser.parse_args()
    
    while args.ode_name not in ["SimpleHO","HenonHeiles","DampedHO"]:
        args.ode_name = input("Type the correct ODE name:\n")
    
    print("Plot Loss : ",args.plot_loss)
    print("Plot errors : ",args.plot_errors)
    print("Plot solutions : ",args.plot_solutions)
    print("Plot long time energy behaviour : ",args.plot_energy)

    model_path = args.save_path + "savedModels/"
    figure_path = args.save_path + 'figures/'
    losses_path = args.save_path + "losses/"
    
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)
    if not os.path.exists(figure_path+"/losses/"):
        os.mkdir(figure_path+"/losses/")
    if not os.path.exists(figure_path+"/errors/"):
        os.mkdir(figure_path+"/errors/")
    if not os.path.exists(figure_path+"/orbits/"):
        os.mkdir(figure_path+"/orbits/")
    if not os.path.exists(figure_path+"/solutions/"):
        os.mkdir(figure_path+"/solutions/")
    if not os.path.exists(figure_path+"/energy/"):
        os.mkdir(figure_path+"/energy/")
    
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
        figure = figure_path,
        losses = losses_path
        )
 
    print(settings.paths['model'])
    
    torch.manual_seed(1)
    np.random.seed(1)
    dtype=torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_parameters = dict(
        hidden_nodes = 10,
        act_name = 'tanh',
        nlayers = 3,
        device=device,
        dtype=dtype,
        d=vec.ndim_total
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
      
    ode_name = args.ode_name 
    if args.plot_errors:
        errors = {
            "pinnReg":[],
            "pinnNoReg":[],
            "hamReg":[],
            "noHamReg":[],
            "mixed":[]
        }
    
    if ode_name=="DampedHO":
        ode_name=ode_name+f"_{str(vec.ll)}"
    
    for name_experiment in ["hamReg","noHamReg","mixed","pinnReg","pinnNoReg"]:
        
        try:
            if args.plot_solutions or args.plot_errors or args.plot_energy:
                
                path = f"{settings.paths['model']}/{ode_name}/{name_experiment}"
                model_path = get_last_trained_model(path) #it appends the latest time stamp
                print(path)
                if name_experiment=="pinn":
                    model = genericNet(model_parameters,vec=vec, dt=training_parameters['dt']) #Usual network
                else:
                    model = sympNet(model_parameters,vec=vec, dt=training_parameters['dt']) #Symplectic network
                model.to(device);

                model.load_state_dict(torch.load(model_path,map_location=device),strict=False)
                print("Model loaded correctly")
                print("Generating the solutions")
                q0,pi0,tf,dtype,device = system_parameters['q0'], system_parameters['pi0'], training_parameters['tf'], training_parameters['dtype'], training_parameters['device']
                vec,t_eval,sol_scipy,sol_slimplectic,sol_network = generate_solutions(vec,q0,pi0,tf,model,dtype,device)

            if args.plot_loss:
                print("Plotting the losses")      
                path = f"{losses_path}{ode_name}/{name_experiment}"
                trainingLossPath,testLossPath = get_last_losses(path)
                print("Training Loss ")
                training_loss = np.loadtxt(trainingLossPath)
                test_loss = np.loadtxt(testLossPath)
                plotLosses(training_loss,test_loss,name_experiment,ode_name)

            if args.plot_energy:
                print("Plotting long time energy behaviour")
                plotLongTimeEnergy(vec,ode_name,name_experiment,t_eval,sol_scipy,sol_network)

            if args.plot_solutions:
                print("Plotting the solutions")
                plotSolutions(vec,ode_name,name_experiment,t_eval,sol_scipy,sol_network,sol_slimplectic=None)
                if vec.ndim_spatial == 1:
                    plotSolutions_2d(vec,ode_name,name_experiment,t_eval,sol_scipy,sol_network,sol_slimplectic=None)

            if args.plot_errors:
                if vec.isa_doubled_variables_system:
                    errors[name_experiment] = np.sqrt(
                        (sol_network[0]-sol_scipy[0])**2 + 
                        (sol_network[2]-sol_scipy[1])**2 
                    )
                else:
                    errors[name_experiment] = np.linalg.norm(sol_network-sol_scipy,axis=0,ord=2)
        except:
            print(f"Saved files for the setting ode_name={ode_name}, name_experiment={name_experiment} not found.\n\n Moving to the next.")
    
    #Out of the for loop since it is plotting all the models together
    if args.plot_errors:
        print("Plotting the errors")
        plotErrors(t_eval,errors,name_experiment,ode_name)