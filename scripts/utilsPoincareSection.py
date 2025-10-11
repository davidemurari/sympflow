import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


from scripts.networks import sympNet, genericNet
from scripts.utils import *
from scripts.training import trainModel, trainingSupervised
from scripts.sampling import sample_ic, generateDataSupervised, createDatasetSupervised
from scripts.plotting import *
from scripts.experiments import *
from scripts.vector_fields import *
import scripts.settings as settings
from tqdm import tqdm
import glob
import time as time_lib

def get_poincare_section(orbit):
    '''
        Takes in an orbit (x,y,x_dot,y_dot) and spits out a Poincaré section (y,y_dot) as a tuple.
    '''
        
    x, y, x_dot, y_dot = orbit[0], orbit[1], orbit[2], orbit[3]

    y_poincare = []
    y_dot_poincare = []
    for i in range(len(x) - 1):
        if (x[i] * x[i + 1] < 0.0):
            y_poincare.append(0.5 * (y[i] + y[i + 1]))
            y_dot_poincare.append(0.5 * (y_dot[i] + y_dot[i + 1]))

    return y_poincare, y_dot_poincare

def get_random_intial_conditons(energy):
    result = False
    while not result:
        with np.errstate(invalid='raise'):
            try:
                q1, q2 = 0.3*(1-2*np.random.random(2))
                p1= 0.2*(1-2*np.random.random())
                p2 = abs(np.sqrt(2*(energy - (q1**2*q2 - q2**3/3.0))-(q1**2+q2**2+p1**2)))
                result = True 
            except FloatingPointError:
                continue 

    initial_state = np.array([q1, q2, p1, p2])
    return initial_state

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
    res = solve_ivp(fun, t_span=t_span, t_eval=t_eval, y0=y0, method='RK45',rtol=1e-12)
    return res.y,res.t

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

def approximate_solution(y0, model, t0, tf, fine_resolution, dtype, device):
    with torch.no_grad():

        d = len(y0)
        dt = model.dt

        tf = (tf//dt)*dt #to make sure we end at a multiple of dt

        num_intervals = int((tf - t0) / dt) 
        ref_interval = torch.linspace(0, dt, fine_resolution, device=device)[1:]  # 10 sub-steps
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