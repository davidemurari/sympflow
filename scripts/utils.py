import torch
import numpy         as np
from torch.func      import jacrev,vmap
from IPython.display import display, Math
from scipy.integrate import solve_ivp
from tqdm import tqdm
import os

from scripts.sampling import sample_ic

import sys


class vecField:
    def __init__(self,system="damped-harmonic-oscillator",d=4):
        
        #This class defines the main properties of the dynamical systems
        
        self.system = system
        self.d = d
        
        #Here we define the parameters of the vector field and say if it will be based
        #on double variables (is_double_variables=True) or only the physical ones (=False)
        if system=="damped-harmonic-oscillator":
            #Model parameters
            self.is_double_variables = True
            self.m  = 1.0
            self.k  = 1.0
            self.ll = 0.01 #Make sure it agrees with ll in experiments.py
                  
        elif system=="harmonic-oscillator":
            #Model parameters
            self.is_double_variables = False
            self.m  = 1.0
            self.k  = 1.0
        
        elif system=="henon-heiles":
            #Model parameters
            self.is_double_variables = False
            self.l = 1.

             
    def eval(self,x,pi):
        
        #This method implements the right-hand-side of the ODEs we work with.
        #It should accomodate batches of inputs, and return a batch of vector fields
        #evaluated. The inputs of the method are (x,pi), where x collects all the components
        #of the configuration variable, and pi all the conjugate momenta.
        
        if self.system=="damped-harmonic-oscillator":
            x1,x2 = x[:,:self.d//4],x[:,self.d//4:]
            pix1,pix2 = pi[:,:self.d//4],pi[:,self.d//4:]
            return torch.cat((pix1/(self.m) + self.ll/(2*self.m)*(x1-x2),
                            -pix2/self.m - self.ll/(2*self.m)*(x1-x2),
                            -self.ll/(2*self.m)*(pix1-pix2) - self.k*x1,
                            self.ll/(2*self.m)*(pix1-pix2) + self.k*x2),dim=1)
        elif self.system=="harmonic-oscillator":
            return torch.cat((pi/self.m,
                              -self.k*x),dim=1)
        elif self.system=="henon-heiles":
            x1 = x[:,0:1]
            x2 = x[:,1:2]
            return torch.cat((pi,
                              -x1-2*x1*x2,
                              -x2-x1**2-x2**2),dim=1)
        else:
            print("Dynamics not implemented")
            
    def residualLoss(self,z,z_d):
        
        #This method computes the residual loss function, which generally is the 
        #mean squared difference between the derivative z_d, computed with automatic
        #differentiation, and the evaluation of the vector field in z.
        
        d = len(z[0])
        
        if self.system=="damped-harmonic-oscillator":
            x = z[:,:d//2]
            pi = z[:,d//2:]
            return torch.mean((self.eval(x,pi)-z_d)**2)
    
        elif self.system=="harmonic-oscillator":
            #Conservative
            x = z[:,:d//2]
            pi  = z[:,d//2:]
            return torch.mean((self.eval(x,pi)-z_d)**2)
    
        elif self.system=="henon-heiles":
            #Conservative
            x = z[:,:d//2]
            pi  = z[:,d//2:]
            return torch.mean((self.eval(x,pi)-z_d)**2)
        else:
            print("Dynamics not implemented")
            

def isSymplectic(model,device,dtype):
    #Checks if the model defines a symplectic map
    #It is better to work in double precision here
    def map(y,t):
        return model(y.reshape(-1,model.d),t.reshape(-1,1))
    
    ## Check if the network is symplectic
    y = torch.randn(1,model.d,dtype=dtype).to(device)
    t = torch.randn(1,1,dtype=dtype).to(device)
    jac = vmap(jacrev(map,argnums=0))(y,t)[0,0]
    id = torch.eye(model.d//2,dtype=dtype)
    zz = torch.zeros((model.d//2,model.d//2),dtype=dtype)
    row1 = torch.cat((zz,id),dim=1)
    row2 = torch.cat((-id,zz),dim=1)
    J = torch.cat((row1,row2),dim=0)
    display(Math(r"\mathcal{N}'(x)^T\mathbb{J}\mathcal{N}'(x):"))
    print((jac.T @ J.to(device) @ jac).detach().cpu().numpy())
    print("Is the network symplectic? ",torch.allclose(jac.T @ J.to(device) @ jac,J.to(device)))

def solution_scipy(y0,t_eval,vec):
    
    #This method computes the approximate solution starting from y0
    #over the time interval t_eval of the vector field vec using a BDF
    #method from the Scipy library.
    
    def fun(t,y):
        q,p = y[:vec.ndim_total//2],y[vec.ndim_total//2:]
        if(not vec.isa_doubled_variables_system):
            q = torch.tensor(q).unsqueeze(0)
            p = torch.tensor(p).unsqueeze(0)
            q_grad,p_grad = vec.eval_vec_field(q,p)
            return np.concatenate([q_grad.squeeze(0).numpy(),p_grad.squeeze(0).numpy()],axis=0)
        elif vec.system_parameters['system']=="damped-harmonic-oscillator": 
            q,p = y[:vec.ndim_total//4],y[vec.ndim_total//4:]
            return np.concatenate([p/vec.m,-vec.ll/vec.m*p-vec.k*q],axis=0)
        else:
            print("Dynamics not implemented")
    
    t_span = [t_eval[0],t_eval[-1]]
    res = solve_ivp(fun, t_span=t_span, t_eval=t_eval, y0=y0, method='RK45').y
    return res

def approximate_solution(y0, model, time, dtype, device):
    with torch.no_grad():

        t0,tf = time[0], time[-1]

        d = len(y0)
        dt = model.dt

        n_steps = int(round(tf / dt))          
        tf      = n_steps * dt                 

        num_intervals = int((tf - t0) / dt) 
        ref_interval = torch.linspace(0, dt, 11, device=device)[1:]  # 10 sub-steps
        jump = len(ref_interval)

        # Total solution storage, starting with initial condition
        sol = torch.zeros((num_intervals * jump + 1, d), dtype=dtype, device=device)
        sol[0] = y0
        time = torch.linspace(t0, tf, jump * num_intervals + 1, device=device)

        # Prepare initial conditions and times for parallel fill-in
        intermediate_sols = torch.zeros((num_intervals, d), device=device)
        intermediate_sols[0] = y0

        # Step 1: Sequentially calculate values at multiples of dt
        for i in range(1, num_intervals):
            intermediate_sols[i] = model(intermediate_sols[i - 1:i], torch.tensor([[dt]], dtype=dtype).to(device))
        # Step 2: Parallel fill for sub-intervals
        times_global = torch.kron(torch.ones(num_intervals, device=device), ref_interval)#.view(-1, 1)
        initial_conditions = intermediate_sols.repeat_interleave(jump, dim=0)
        # Perform parallel model predictions and store in sol
        predictions = model(initial_conditions, times_global)
        sol[1:] = predictions  # Fill the solution with parallel predictions directly
    return sol, time

def generate_solutions(vec,q0,pi0,tf,model,dtype,device):
    y0 = torch.cat((q0,pi0))
        
    if vec.isa_doubled_variables_system:
        y0_np = torch.cat((q0[:vec.ndim_total//4],pi0[:vec.ndim_total//4])).detach().cpu().numpy()
    else: y0_np = y0.detach().cpu().numpy()
   
    t_eval = np.linspace(0,tf,int(tf*10+1)) 

    sol_network, t_eval = approximate_solution(y0.to(device),model,t_eval,dtype,device)
    t_eval = t_eval.detach().cpu().numpy()
    sol_network = sol_network.detach().cpu().numpy().T.squeeze()
    sol_scipy = solution_scipy(y0_np,t_eval=t_eval,vec=vec)

    sol_slimplectic = None
    if(sol_slimplectic is not None): 
        sol_slimplectic=np.array(sol_slimplectic).squeeze()
    return vec,t_eval,sol_scipy,sol_slimplectic,sol_network


def generate_test_set_unsupervised(args,system_parameters,training_parameters,vec):
    if not os.path.exists("testSet"):
        os.mkdir("testSet")
    if args.ode_name=="DampedHO":
        
        if not os.path.exists(
            f"testSet/{system_parameters['vec_field_name']}_{str(vec.ll)}_z.csv"
        ) or not os.path.exists(f"testSet/{system_parameters['vec_field_name']}_{str(vec.ll)}__t.csv"):
            
            print("Generating the test initial conditions...")
            
            z, t = sample_ic(
                vec.system_parameters,
                vec,
                dtype=training_parameters["dtype"],
                n_samples=training_parameters["n_test"],
                dt=training_parameters["dt"],
                factor=1,
                t0=training_parameters["t0"],
            )
            z = z.detach().cpu().numpy()
            t = t.detach().cpu().numpy()
            np.savetxt(
                f"testSet/{system_parameters['vec_field_name']}_{str(vec.ll)}_z.csv",
                z.reshape(-1),
                delimiter=",",
            )
            np.savetxt(
                f"testSet/{system_parameters['vec_field_name']}_{str(vec.ll)}_t.csv",
                t.reshape(-1),
                delimiter=",",
            )

        z_test = np.loadtxt(f"testSet/{system_parameters['vec_field_name']}_{str(vec.ll)}_z.csv").reshape(
            -1, vec.ndim_total
        )
        t_test = np.loadtxt(f"testSet/{system_parameters['vec_field_name']}_{str(vec.ll)}_t.csv").reshape(
            -1, 1
        )
    else:
        if not os.path.exists(
        f"testSet/{system_parameters['vec_field_name']}_z.csv"
            ) or not os.path.exists(f"testSet/{system_parameters['vec_field_name']}_t.csv"):
            
            print("Generating the test initial conditions...")
            
            z, t = sample_ic(
                vec.system_parameters,
                vec,
                dtype=training_parameters["dtype"],
                n_samples=training_parameters["n_test"],
                dt=training_parameters["dt"],
                factor=1,
                t0=training_parameters["t0"],
            )
            z = z.detach().cpu().numpy()
            t = t.detach().cpu().numpy()
            np.savetxt(
                f"testSet/{system_parameters['vec_field_name']}_z.csv",
                z.reshape(-1),
                delimiter=",",
            )
            np.savetxt(
                f"testSet/{system_parameters['vec_field_name']}_t.csv",
                t.reshape(-1),
                delimiter=",",
            )

        z_test = np.loadtxt(f"testSet/{system_parameters['vec_field_name']}_z.csv").reshape(
            -1, vec.ndim_total
        )
        t_test = np.loadtxt(f"testSet/{system_parameters['vec_field_name']}_t.csv").reshape(
            -1, 1
        )
    print("Test initial conditions generated.\n")  
    return z_test, t_test

def generate_missing_directories(model_path,losses_path,ode_name,name_experiment,settings,args):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(f"{model_path}{ode_name}"):
        os.mkdir(f"{model_path}{ode_name}")
    if not os.path.exists(f"{model_path}{ode_name}/{name_experiment}"):
        os.mkdir(f"{model_path}{ode_name}/{name_experiment}")
    if not os.path.exists(settings.paths["model"] + f"tmpFiles/"):
        os.mkdir(settings.paths["model"] + f"tmpFiles/")

    # These are to save the losses
    if not os.path.exists(losses_path):
        os.mkdir(losses_path)
    if not os.path.exists(f"{losses_path}{ode_name}"):
        os.mkdir(f"{losses_path}{ode_name}")
    if not os.path.exists(f"{losses_path}{ode_name}/{name_experiment}"):
        os.mkdir(f"{losses_path}{ode_name}/{name_experiment}")