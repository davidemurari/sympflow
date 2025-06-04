import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time as time_lib
import matplotlib

from scripts.networks import *
from scripts.vector_fields import *
from scripts.utils import *
from scripts.settings import *
from scripts.experiments import *

#Fixing the configuration for the plots
plt.rcParams["figure.dpi"] = 300
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10

cwd = os.getcwd()
torch.manual_seed(1)
np.random.seed(1)
dtype=torch.float32
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def plotLosses(training_loss,test_loss,name_experiment,ode_name,title_fig=None):
    
    path = "unsupervisedNetworks/figures/"
    colors = ["k","b","darkgreen","darkslategrey"]

    plt.rcParams["figure.figsize"] = (3,2)

    fig = plt.figure()
    epochs = len(training_loss)
    epoch_list = np.arange(epochs)
    if epochs>1000:
        indices_to_consider = np.arange(0,epochs,100)
    else:
        indices_to_consider = np.arange(epochs)
    training_loss = training_loss[indices_to_consider]
    test_loss = test_loss[indices_to_consider]
    
    plt.loglog(epoch_list[indices_to_consider],training_loss,'-',c=colors[0])
    plt.loglog(epoch_list[0],training_loss[0],'-',c=colors[0],label="Training Loss")
    
    plt.loglog(epoch_list[indices_to_consider],test_loss,'-',c=colors[1])
    plt.loglog(epoch_list[0],test_loss[0],'-',c=colors[1],label="Test Loss")

    timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    if name_experiment=="pinnReg":
        title = "MLP with regularization"
    elif name_experiment=="pinnNoReg":
        title = "MLP just residual"
    elif name_experiment=="hamReg":
        title = r"\texttt{SympFlow} with regularization"
    elif name_experiment=="noHamReg":
        title = r"\texttt{SympFlow} just residual"
    elif name_experiment=="mixed":
        title = r"\texttt{SympFlow} with mixed training"
    plt.title(f"Training and test loss: {title}")
    if title_fig==None:
        plt.savefig(f"{path}/losses/losses_{name_experiment}_{ode_name}_{timestamp}.pdf",bbox_inches='tight')
    else:
        plt.savefig(f"{path}/losses/{title_fig}.pdf",bbox_inches='tight')
    

def plotErrors(t_eval,errors,name_experiment,ode_name,title_fig=None):
    
    plt.rcParams["figure.figsize"] = (3,2)
    
    figure_path = "unsupervisedNetworks/figures/"
    colors = ["k","b","darkgreen","red","cyan"]
    
    if (len(errors["pinnReg"])>0 or 
        len(errors["pinnNoReg"])>0 or 
        len(errors["hamReg"])>0 or 
        len(errors["noHamReg"])>0 or 
        len(errors["mixed"])>0):
            
        fig = plt.figure()
        
        if len(errors["pinnReg"])>0:
            plt.loglog(t_eval,errors["pinnReg"],'-',c=colors[0])
            plt.loglog(t_eval[0],errors["pinnReg"][0],'-',c=colors[0],label="MLP with regularization")
        
        if len(errors["pinnNoReg"])>0:
            plt.loglog(t_eval,errors["pinnNoReg"],'-',c=colors[1])
            plt.loglog(t_eval[0],errors["pinnNoReg"][0],'-',c=colors[1],label="MLP just residual")
        
        if len(errors["hamReg"])>0:
            plt.loglog(t_eval,errors["hamReg"],'-',c=colors[2])
            plt.loglog(t_eval[0],errors["hamReg"][0],'-',c=colors[2],label="SympFlow with regularization")
        
        if len(errors["noHamReg"])>0:
            plt.loglog(t_eval,errors["noHamReg"],'-',c=colors[3])
            plt.loglog(t_eval[0],errors["noHamReg"][0],'-',c=colors[3],label="SympFlow just residual")
        
        if len(errors["mixed"])>0:
            plt.loglog(t_eval,errors["mixed"],'-',c=colors[4])
            plt.loglog(t_eval[0],errors["mixed"][0],'-',c=colors[4],label="SympFlow with mixed training procedure")
        
        timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 
        
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\left\|\mathbf{z}_{\text{ref}}(t)-\mathbf{z}_{\text{net}}(t)\right\|_2$")
        plt.legend(ncol=2,bbox_to_anchor=(0.5,-0.3),loc='upper center')
        plt.title("Comparison of the errors")
        if title_fig==None:
            plt.savefig(f"{figure_path}/errors/errors_{ode_name}_{timestamp}.pdf",bbox_inches='tight')
        else:
            plt.savefig(f"{figure_path}/errors/{title_fig}",bbox_inches='tight')
        
    else:
        pass

def plotSolutions_2d(vec,ode_name,name_experiment,t_eval,sol_scipy,sol_network,sol_slimplectic=None,is_supervised=False,figure_path = "unsupervisedNetworks/figures/",title_fig=None):
        
    plt.rcParams["figure.figsize"] = (2,2)
    back_colors = ["k","b","darkgreen","darkslategrey"]
    front_colors = ["r","plum","darkturquoise","lightsalmon"]
        
    suffix = ["x","y","z"]
    
    if is_supervised:

        if name_experiment=="pinn":
            title = "MLP"
        elif name_experiment=="sympflow":
            title = r"\texttt{SympFlow}"
    else:
        if name_experiment=="pinnReg":
            title = "MLP with regularization"
        elif name_experiment=="pinnNoReg":
            title = "MLP just residual"
        elif name_experiment=="hamReg":
            title = r"\texttt{SympFlow} with regularization"
        elif name_experiment=="noHamReg":
            title = r"\texttt{SympFlow} just residual"
        elif name_experiment=="mixed":
            title = r"\texttt{SympFlow} with mixed training"
    
    fig = plt.figure()
 
    second_component = 2 if vec.isa_doubled_variables_system else 1

    # Plot for q variables
    plt.plot(sol_scipy[0],sol_scipy[1], '-', c=back_colors[0], label=rf"ODE45")
    plt.plot(sol_network[0],sol_network[second_component], '--', c=front_colors[0], label="Network")

    # Add labels and legend
    plt.xlabel(r"$q$")
    plt.ylabel(r"$p$")
    plt.legend()

    timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 

    # Add a common title for the entire figure
    plt.title(title)
    if title_fig==None:
        plt.savefig(f"{figure_path}/orbits/orbit_{name_experiment}_{ode_name}_{timestamp}.pdf",bbox_inches='tight')
    else:
        plt.savefig(f"{figure_path}/orbits/{title_fig}.pdf",bbox_inches='tight')
    

def plotSolutions(vec,ode_name,name_experiment,t_eval,sol_scipy,sol_network,sol_slimplectic=None,is_supervised=False,figure_path = "unsupervisedNetworks/figures/",title_fig=None):
        

    plt.rcParams["figure.figsize"] = (6, 4.5)

    d = vec.ndim_spatial
    
    back_colors = ["k","b","darkgreen","darkslategrey"]
    front_colors = ["r","plum","darkturquoise","lightsalmon"]
    
    #fig = plt.figure(figsize=(20,10))
    
    suffix = ["x","y","z"]
    
    if is_supervised:
        if name_experiment=="pinn":
            title = "MLP"
        elif name_experiment=="sympflow":
            title = r"\texttt{SympFlow}"
    else:
        if name_experiment=="pinnReg":
            title = "MLP with regularization"
        if name_experiment=="pinnNoReg":
            title = "MLP just residual"
        elif name_experiment=="hamReg":
            title = r"\texttt{SympFlow} with regularization"
        elif name_experiment=="noHamReg":
            title = r"\texttt{SympFlow} just residual"
        elif name_experiment=="mixed":
            title = r"\texttt{SympFlow} with mixed training"
    
    fig = plt.figure()
    gs = fig.add_gridspec(2, d, hspace=0.55, wspace=0.4)  # 2 rows, d//2 columns

    fact = 2 if vec.isa_doubled_variables_system else 1

    for i in range(d):
        
        ax = fig.add_subplot(gs[0, i])

        # Plot for q variables
        ax.plot(t_eval[0], sol_scipy[i, 0], '-', c=back_colors[i], label=rf"$q_{{{suffix[i]}}}$ ODE45")
        ax.plot(t_eval, sol_scipy[i], '-', c=back_colors[i])
        ax.plot(t_eval[0], sol_network[i, 0], '--', c=front_colors[i], label=rf"$q_{{{suffix[i]}}}$ Network")
        ax.plot(t_eval, sol_network[i], '--', c=front_colors[i])

        # Add labels and legend
        ax.set_xlabel(r"$t$")
        ax.legend(loc='center',bbox_to_anchor=(0.5,1.15),ncol=2)

        ax = fig.add_subplot(gs[1, i])

        # Plot for pi variables
        ax.plot(t_eval[0], sol_scipy[d + i, 0], '-', c=back_colors[d + i], label=rf"$p_{{{suffix[i]}}}$ ODE45")
        ax.plot(t_eval, sol_scipy[d + i], '-', c=back_colors[d + i])
        ax.plot(t_eval[0], sol_network[int(fact*d + i), 0], '--', c=front_colors[d + i], label=rf"$p_{{{suffix[i]}}}$ Network")
        ax.plot(t_eval, sol_network[int(fact*d + i)], '--', c=front_colors[d + i])

        # Add labels and legend
        ax.set_xlabel(r"$t$")
        ax.legend(loc='center',bbox_to_anchor=(0.5,1.15),ncol=2)

    timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 

    # Add a common title for the entire figure
    plt.suptitle(title,y=1.02)
    if title_fig==None:
        plt.savefig(f"{figure_path}/solutions/solutions_{name_experiment}_{ode_name}_{timestamp}.pdf",bbox_inches='tight')
    else:
        plt.savefig(f"{figure_path}/solutions/{title_fig}.pdf",bbox_inches='tight')

def plotLongTimeEnergy(vec,ode_name,name_experiment,t_eval,sol_scipy,sol_network,sol_slimplectic=None,is_supervised=False,figure_path = "unsupervisedNetworks/figures/",title_fig=None):
    
    plt.rcParams["figure.figsize"] = (3,2)

    d = vec.ndim_spatial
    factor = 2 if vec.isa_doubled_variables_system else 1
   
    if is_supervised:
        if name_experiment=="pinn":
            title = "MLP"
        elif name_experiment=="sympflow":
            title = r"\texttt{SympFlow}"
    else:
        if name_experiment=="pinnReg":
            title = "MLP with regularization"
        elif name_experiment=="pinnNoReg":
            title = "MLP just residual"
        elif name_experiment=="hamReg":
            title = r"\texttt{SympFlow} with regularization"# with Hamiltonian Regularization"
        elif name_experiment=="noHamReg":
            title = r"\texttt{SympFlow}  just residual"
        elif name_experiment=="mixed":
            title = r"\texttt{SympFlow} with mixed training"
    fig = plt.figure()
    
    q0 = sol_scipy[:d,0]
    pi0 = sol_scipy[d:,0]
    
    q_scipy = sol_scipy.T[:,:d]
    pi_scipy = sol_scipy.T[:,d:]
    
    if vec.isa_doubled_variables_system:
        q0 = np.concatenate((q0,q0)).reshape(1,-1)
        pi0 = np.concatenate((pi0,-pi0)).reshape(1,-1)
        q_scipy = np.concatenate((sol_scipy.T[:,:d],sol_scipy.T[:,:d]),axis=1)
        pi_scipy = np.concatenate((sol_scipy.T[:,d:],-sol_scipy.T[:,d:]),axis=1)

    E0 = vec.eval_hamiltonian(q0,pi0).reshape(-1)

    plt.loglog(t_eval,np.abs(vec.eval_hamiltonian(q_scipy,pi_scipy).reshape(-1)-E0),'r-',label="Energy ODE45")
    plt.loglog(t_eval,np.abs(vec.eval_hamiltonian(sol_network.T[:,:factor*d],sol_network.T[:,factor*d:]).reshape(-1)-E0),'c--',label="Energy Network")

    plt.legend()
    plt.ylabel(r"$|H(\psi_t(z_0))-H(z_0)|$")
    plt.xlabel(r"$t$")
    plt.title(title)
    timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 
    
    if title_fig==None:
        print(figure_path)
        plt.savefig(f"{figure_path}/energy/longEnergy_{name_experiment}_{ode_name}_{timestamp}.pdf",bbox_inches='tight')
    else:
        print(figure_path)
        plt.savefig(f"{figure_path}/energy/{title_fig}.pdf",bbox_inches='tight')

    #Plot the physical energy
    if vec.isa_doubled_variables_system:
        fig = plt.figure()
        E_ode45 = vec.physical_energy(sol_scipy.T[:,:d],sol_scipy.T[:,d:]).reshape(-1)
        E_network = vec.physical_energy(sol_network.T[:,:d],sol_network.T[:,2*d:2*d+d]).reshape(-1)
        difference = np.abs(E_network-E_ode45) / np.diff(t_eval)[0]
        plt.loglog(t_eval,difference,'r-',label="Relative energy variation")
        plt.legend()
        plt.ylabel(r"$E(\psi_t(z_0))$")
        plt.xlabel(r"$t$")
        plt.title(title)
        timestamp = time_lib.strftime("%Y%m%d_%H%M%S") 
        
        if title_fig==None:
            print(figure_path)
            plt.savefig(f"{figure_path}/energy/physicalEnergy_{name_experiment}_{ode_name}_{timestamp}.pdf",bbox_inches='tight')
        else:
            print(figure_path)
            plt.savefig(f"{figure_path}/energy/{title_fig}.pdf",bbox_inches='tight')