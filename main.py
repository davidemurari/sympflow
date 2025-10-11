import time as time_lib
import torch
import argparse
import warnings

from scripts.networks import sympNet, genericNet

from scripts.utils import *
from scripts.training import trainModel
from scripts.sampling import sample_ic
from scripts.experiments import *
from scripts.vector_fields import *
import scripts.settings as settings

import glob
import numpy as np
import os
from datetime import datetime

def get_last_trained_model(path):
    # Get the list of all files matching the pattern
    files = glob.glob(os.path.join(path, "trained_model_*.pt"))
    # Function to extract the timestamp from the filename
    try:
        
        def extract_timestamp(filename):
            basename = os.path.basename(filename)
            timestamp_str = basename[len("trained_model_"):-len(".pt")]
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

        # Find the latest file by comparing timestamps
        latest_file_name = max(files, key=extract_timestamp)
        return latest_file_name

    except:
        return None

if __name__ == "__main__":
    cwd = os.getcwd()

    torch.manual_seed(1)
    np.random.seed(1)
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Training models")
    parser.add_argument("--save_path", default="unsupervisedNetworks/")
    parser.add_argument("--ode_name", default="DampedHO")  # among HenonHeiles,SimpleHO,DampedHO
    parser.add_argument("--final_time", default=10.0, type=float)  # for the temporary plots while training
    parser.add_argument("--number_layers", default=3, type=int)
    parser.add_argument("--epochs", default=50001, type=int)  # number of training epochs
    parser.add_argument("--dt", default=1.0, type=float)  # during the training procedure
    parser.add_argument("--name_experiment", default="pinnReg")  # among pinnReg,pinnNoReg,mixed,hamReg,noHamReg
    args = parser.parse_args()
    
    if not os.path.exists(f"{args.save_path}"):
        os.mkdir(f"{args.save_path}")
    
    print("\n\n================================================")
    print(f"Name of the considered ODE: {args.ode_name}")
    print(f"Delta t={args.dt}")
    print(f"Model under consideration: {args.name_experiment}")
    print(f"Number of training epochs: {args.epochs}")
    print(f"Number of layers: {args.number_layers}")
    print("=================================================\n\n")

    # Domain details
    # All the systems from experiments.py
    
    while args.ode_name not in ["SimpleHO","HenonHeiles","DampedHO"]:
     args.ode_name = input("Type the correct ODE name:\n")
     if args.ode_name in ["SimpleHO","HenonHeiles","DampedHO"]:
        print(f"\nName of the considered ODE: {args.ode_name}\n")
    while args.name_experiment not in ["pinnNoReg","pinnReg","mixed","hamReg","noHamReg"]:
     args.name_experiment = input("Type the correct model name:\n")     
     if args.name_experiment in ["pinnNoReg","pinnReg","mixed","hamReg","noHamReg"]:
        print(f"\n Model under consideration: {args.name_experiment}\n")
    
    if args.ode_name == "SimpleHO":
        system_parameters = SimpleHO_exp
    elif args.ode_name == "DampedHO":
        system_parameters = DampedHO_exp
    elif args.ode_name == "HenonHeiles":
        system_parameters = Henon_Heiles_exp
        
   
    # Initialise the vector field class according to the experiment
    print(system_parameters["vec_field_name"])
    vector_field_class = globals()[system_parameters["vec_field_name"]]

    # Initialise the vector field, network and optimizer
    vec = vector_field_class(system_parameters)

    # Define the model parameters
    model_parameters = dict(
        hidden_nodes=10,
        act_name="tanh",
        nlayers=args.number_layers,
        device=device,
        dtype=dtype,
        d=vec.ndim_total
    )
    
    ode_name = args.ode_name   
    if args.ode_name == "DampedHO":
        ode_name = "DampedHO_"+ str(vec.ll) 

    name_experiment = args.name_experiment

    ### Paths to save models and figures
    model_path = args.save_path + "savedModels/"
    figure_path = args.save_path + f"tempFigs/{ode_name}/{name_experiment}/"
    losses_path = args.save_path + "losses/"

    # dictionary of paths
    settings.paths = dict(model=model_path, figure=figure_path, losses=losses_path)

    path = f"{settings.paths['model']}/{ode_name}/hamReg"
    path_trained_hamReg = get_last_trained_model(path)
    print(path_trained_hamReg) 
    
    if name_experiment == "mixed" and path_trained_hamReg==None:
        print("Not found the model trained with hamReg")
        print("We thus replace the name_experiment to hamReg")
        name_experiment = "hamReg"

    ### Paths to save models and figures
    figure_path = (args.save_path + f"tempFigs/{ode_name}/{name_experiment}/")
    settings.paths = dict(model=model_path, figure=figure_path, losses=losses_path)

    training_parameters = dict(
        dt= args.dt,
        tf= args.final_time,
        t0=- 0.1,
        n_train= 700,
        n_test= 5000,
        epochs= args.epochs,
        device= device,
        dtype= dtype,
        lr= 5e-3,
        name_run= name_experiment,
        hamReg= args.name_experiment=="hamReg" or args.name_experiment=="pinnReg"
    )
    
    if args.ode_name == "DampedHO": 
        HO_freq       = np.sqrt(vec.k/vec.m)
        damping_ratio = vec.ll/np.sqrt( vec.ll**2+ HO_freq**2)
        Damping_freq  = HO_freq*np.sqrt(1-damping_ratio)
        Decay_time    = 1/(HO_freq*damping_ratio)
        final_time    = Decay_time
        
        print("\n ==== RUN CONFIGURATION =====")
        print("dt = ", args.dt)
        print("System :", name_experiment)
        print("ODE :",ode_name )
        print("k, ll, m :",vec.k, vec.ll,vec.m)
        print("HO_freq :", HO_freq)
        print("Damping Ratio :", damping_ratio)
        print("Decay time :",Decay_time)
        print("Damping_freq: ", Damping_freq)
        print("=========================== \n")

    # Check if the test set is available
    # We use this to compare all the models on the same test set of initial conditions
    z_test, t_test = generate_test_set_unsupervised(args,system_parameters,training_parameters,vec)

    if dtype == torch.float32:
        z_test = torch.from_numpy(z_test.astype(np.float32)).to(device)
        t_test = torch.from_numpy(t_test.astype(np.float32)).to(device)
    else:
        z_test = torch.from_numpy(z_test).to(device)
        t_test = torch.from_numpy(t_test).to(device)

    test_set = dict(z=z_test, t=t_test)

    if name_experiment == "pinnReg" or name_experiment == "pinnNoReg":
        model = genericNet(
            model_parameters, vec=vec, dt=training_parameters["dt"]
        )  # Usual network
    else:
        model = sympNet(
            model_parameters, vec=vec, dt=training_parameters["dt"]
        )  # Symplectic network
    model.to(device)
    if name_experiment == "mixed":
        model.load_state_dict(torch.load(path_trained_hamReg, map_location=device))

    # Create folders where we save the results in case they do not exist
    generate_missing_directories(model_path,losses_path,ode_name,name_experiment,settings,args)

    # Adam should be used at first
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_parameters["lr"], weight_decay=0.0
    )

    timestamp = time_lib.strftime("%Y%m%d_%H%M%S")

    print("Training the model...")
    initial_training_time = time_lib.time()
    Loss = trainModel(
        model,
        training_parameters,
        vec,
        system_parameters,
        optimizer,
        test_set=test_set,
        ode_name=ode_name
    )
    final_training_time = time_lib.time()
    model.training_time = (final_training_time-initial_training_time) / args.epochs
    timestamp = time_lib.strftime("%Y%m%d_%H%M%S")

    torch.save(
        model.state_dict(),
        settings.paths["model"]
        + f"{ode_name}/{name_experiment}/trained_model_{timestamp}.pt",
    )
    
    ## Save model
    torch.save(
        model.state_dict(),
        settings.paths["model"]
        + f"{ode_name}/{name_experiment}/trained_model_{timestamp}.pt",
    )
    
    ## Training time
    if not os.path.isfile("unsupervisedNetworks/timingsTrainingPerEpoch.txt"):
        open("unsupervisedNetworks/timingsTrainingPerEpoch.txt", "w").close()
    with open("unsupervisedNetworks/timingsTrainingPerEpoch.txt", "a") as myfile:
        text = f"{timestamp}, {ode_name}, {name_experiment}, number_layers={args.number_layers}, num_epochs={args.epochs}: {model.training_time}\n"
        myfile.write(text)
    
    ##Inference time
    print("Testing the average inference cost")
    initial_time_solutions = time_lib.time()
    y0, _ = sample_ic(vec.system_parameters, vec, dtype, n_samples=100, dt=args.dt, factor=1.1, t0=0.)
    for i in tqdm(range(len(y0))):
        _,_ = approximate_solution(y0[i], model, time=[0,100], dtype=dtype, device=device)
                
    final_time_solutions = time_lib.time()
    model.inference_time = (final_time_solutions-initial_time_solutions) / (args.final_time * len(y0)) #average per 100 initial conditions and over tf.
    
    if not os.path.isfile("unsupervisedNetworks/timingsInferecePer1s.txt"):
            open("unsupervisedNetworks/timingsInferecePer1s.txt", "w").close()
    with open("unsupervisedNetworks/timingsInferecePer1s.txt", "a") as myfile:
        text = f"{timestamp}, {ode_name}, {name_experiment}: {model.inference_time}\n"
        myfile.write(text)