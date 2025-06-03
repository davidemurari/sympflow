import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time as time_lib
import matplotlib.pyplot as plt


from scripts.networks import sympNet, genericNet
from scripts.utils import *
from scripts.training import trainModel, trainingSupervised
from scripts.sampling import generateDataSupervised, createDatasetSupervised
from scripts.experiments import *
from scripts.vector_fields import *
import scripts.settings as settings
import argparse

parser = argparse.ArgumentParser(description="Supervised experiment")
parser.add_argument("--ode_name", default="SimpleHO") #among HenonHeiles,SimpleHO,DampedHO
parser.add_argument("--dt", default=1.0, type=float)
parser.add_argument("--N", default=100, type=int)
parser.add_argument("--M", default=50, type=int)
parser.add_argument("--epsilon", default=0.0, type=float)
parser.add_argument("--name_experiment", default="sympflow") #choose either pinn or sympflow
parser.add_argument("--epochs", default=500, type=int)

args = parser.parse_args()

print("\n\n================================================")
print(f"Name of the considered ODE: {args.ode_name}")
print(f"Delta t={args.dt}")
print(f"Number initial conditions: N={args.N}")
print(f"Number sampled points per initial condition: M={args.M}")
print(f"Standard deviation for the noise: epsilon={args.epsilon}")
print(f"Model under consideration: {args.name_experiment}")
print(f"Number of training epochs: {args.epochs}")
print("=================================================\n\n")

while args.ode_name not in ["SimpleHO","HenonHeiles","DampedHO"]:
     args.ode_name = input("Type the correct ODE name:\n")
     if args.ode_name in ["SimpleHO","HenonHeiles","DampedHO"]:
        print(f"\nName of the considered ODE: {args.ode_name}\n")
while args.name_experiment not in ["pinn","sympflow"]:
     args.name_experiment = input("Type the correct model name:\n")     
     if args.name_experiment in ["pinn","sympflow"]:
        print(f"\n Model under consideration: {args.name_experiment}\n")

ode_name = args.ode_name
dtype = torch.float32

if ode_name=="DampedHO":
    system_parameters = DampedHO_exp
elif ode_name=="HenonHeiles":
    system_parameters = Henon_Heiles_exp
elif ode_name=="SimpleHO":
        system_parameters = SimpleHO_exp 

vector_field_class = globals()[system_parameters["vec_field_name"]]
vec = vector_field_class(system_parameters)

N = args.N #number of initial conditions
M = args.M #number of sampled points on their trajectories
epsilon = args.epsilon #standard deviation of the noise
print("Generating the data...")
sol,time_instants = generateDataSupervised(vec,system_parameters,dtype,N,M,epsilon)
print("Dataset generated.\n")

_,d,_ = sol.shape

sol_flat = np.zeros((N*(M-1),2,d))
t_flat = np.zeros((N*(M-1),1))
for i in range(N):
    for j in range(M-1):
        index = i * (M - 1) + j 
        sol_flat[index,0],sol_flat[index,1] = sol[i,:,0],sol[i,:,j+1]
        t_flat[index] = time_instants[i,j]

dataset = createDatasetSupervised(sol_flat, t_flat)
batch_size = 512
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    tf=10,
    t0=0.,
    n_train=700,
    n_test=5000,
    epochs=args.epochs,
    device=device,
    dtype=dtype,
    lr=1e-4,
)

name_experiment = args.name_experiment

if name_experiment=="pinn":
    model = genericNet(
            model_parameters, vec=vec, dt=training_parameters["dt"]
        )
else:
    model = sympNet(
            model_parameters, vec=vec, dt=training_parameters["dt"]
        )
model.to(device);

steps = len(dataloader)*training_parameters["epochs"]

max_lr = 5e-2
min_lr = 1e-4

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=training_parameters["lr"])

scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            div_factor= max_lr / min_lr,
            total_steps=steps,
            steps_per_epoch=len(dataloader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

print("\nTraining the model...")
trainingSupervised(
    model,
    vec,
    num_epochs=training_parameters["epochs"],
    trainloader=dataloader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device=device
    )

if ode_name=="DampedHO":
    ode_name=ode_name+f"_{str(vec.ll)}"

timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 


if not os.path.exists("supervisedNetworks"):
        os.mkdir("supervisedNetworks")
if not os.path.exists("supervisedNetworks/savedModels"):
        os.mkdir("supervisedNetworks/savedModels")
if not os.path.exists(f"supervisedNetworks/savedModels/{ode_name}"):
        os.mkdir(f"supervisedNetworks/savedModels/{ode_name}")
if name_experiment=="pinn":
    if not os.path.exists(f"supervisedNetworks/savedModels/{ode_name}/pinn"):
            os.mkdir(f"supervisedNetworks/savedModels/{ode_name}/pinn")
else:
    if not os.path.exists(f"supervisedNetworks/savedModels/{ode_name}/sympflow"):
        os.mkdir(f"supervisedNetworks/savedModels/{ode_name}/sympflow")

if not os.path.exists(f"supervisedNetworks/trainingData"):
            os.mkdir(f"supervisedNetworks/trainingData")
plt.savefig(f"supervisedNetworks/trainingData/training_data_{ode_name}_{timestamp}.pdf",bbox_inches="tight")

if name_experiment=="pinn":
    torch.save(
            model.state_dict(),
            f"supervisedNetworks/savedModels/{ode_name}/pinn/trained_model_{timestamp}.pt",
        )
else:
    torch.save(
            model.state_dict(),
            f"supervisedNetworks/savedModels/{ode_name}/sympflow/trained_model_{timestamp}.pt",
        )