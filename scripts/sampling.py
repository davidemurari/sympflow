import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp

torch.set_printoptions(precision=10)

def sample_ic(system_parameters, vec, dtype, n_samples, dt, factor, t0):
    qub = system_parameters["qub"]
    qlb = system_parameters["qlb"]
    piub = system_parameters["qub"]
    pilb = system_parameters["qlb"]

    t = (
        torch.rand((n_samples, 1), dtype=dtype) * dt * factor + t0
    )  # Uniform random initial condition
    
    if vec.isa_doubled_variables_system:
        q1 = torch.rand((n_samples, vec.ndim_spatial), dtype=dtype) * (qub - qlb) + qlb
        q2 = q1 + torch.rand_like(q1)*0.02-0.01

        pi1 = (
            torch.rand((n_samples, vec.ndim_spatial), dtype=dtype) * (piub - pilb)
            + pilb
        )
        pi2 = pi1 + torch.rand_like(pi1)*0.02-0.01
        return torch.cat((q1, q2, pi1, -pi2), dim=1), t
    else:
        q = torch.rand((n_samples, vec.ndim_spatial), dtype=dtype) * (qub - qlb) + qlb
        pi = (
            torch.rand((n_samples, vec.ndim_spatial), dtype=dtype) * (piub - pilb)
            + pilb
        )
        return torch.cat((q, pi), dim=1), t

def generateDataSupervised(vec,system_parameters,dtype,N=1000,M=5,epsilon=0.):

    qub = system_parameters["qub"]
    qlb = system_parameters["qlb"]
    piub = system_parameters["qub"]
    pilb = system_parameters["qlb"]


    def fun(t,y):
        q,p = y[:vec.ndim_total//2],y[vec.ndim_total//2:]
        q = torch.tensor(q).unsqueeze(0)
        p = torch.tensor(p).unsqueeze(0)
        q_grad,p_grad = vec.eval_vec_field(q,p)
        return np.concatenate([q_grad.squeeze(0).numpy(),p_grad.squeeze(0).numpy()],axis=0)
        

    time_instants = 1.1*torch.sort(torch.rand(N,M), dim=1)[0]
    
    q = torch.rand((N, vec.ndim_spatial), dtype=dtype) * (qub - qlb) + qlb
    p = torch.rand((N, vec.ndim_spatial), dtype=dtype) * (piub - pilb) + pilb
    
    if vec.isa_doubled_variables_system:
        q = torch.cat((q,q),dim=1)
        p = torch.cat((p,-p),dim=1)
        
    initial_conditions = torch.cat((q,p),dim=1)
    
    sol = np.zeros((N,vec.ndim_total,M+1))
    sol[:,:,0] = initial_conditions
    for i in range(N):
        sol[i,:,1:] = solve_ivp(
            fun, 
            t_span=[0, time_instants[i,-1]], 
            t_eval=time_instants[i,:], 
            y0=initial_conditions[i], 
            method='RK45',
            atol=1e-10,
            rtol=1e-10).y
        if epsilon>0:
            noise = epsilon * np.random.randn(*sol[i,:,1:].shape)
            sol[i,:,1:] += noise
    return sol,time_instants

class createDatasetSupervised(Dataset):
    def __init__(self, res, t):
        self.x = torch.from_numpy(res[:, 0].astype(np.float32))
        self.t = torch.from_numpy(t.astype(np.float32))
        self.y = torch.from_numpy(res[:, 1].astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.y[idx]