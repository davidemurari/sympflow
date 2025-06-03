import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap

# Custom activation functions
class sinAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class swishAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x

class integralTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(torch.cosh(x))  # this is the primitive of tanh(x)


class sympNet(nn.Module):
    def __init__(
        self, model_parameters = dict(
            hidden_nodes = 100,
            act_name = 'sin',
            nlayers = 3,
            dtype = torch.float,
            d = 4),vec = None, dt = 1.):
        super().__init__()

        torch.manual_seed(1)
        np.random.seed(1)
        
        ## Unpack model_parameters
        neurons, act_name, nlayers, dtype, d = model_parameters['hidden_nodes'], model_parameters['act_name'], model_parameters['nlayers'], model_parameters['dtype'], model_parameters['d']
        # Assert that network dimension is compatible with vec dimension
        if(vec is not None): assert d == vec.ndim_total
            
        self.dtype = dtype
        self.vec = vec
        self.dt = dt

        activations = {
            "tanh": nn.Tanh(),
            "sin": sinAct(),
            "sigmoid": nn.Sigmoid(),
            "swish": swishAct(),
            "intTanh": integralTanh(),
        }

        self.act = activations[act_name]

        self.H = neurons
        self.nlayers = nlayers
        
        #### The dimensions here referes explicitly to the dimensions as seen by the network, i.e. the dimension of variable q
        #### So, if ndim_spatial = d and it is not double variables, then self.d = d, if it is double variables, then self.d = 2*d 
        self.d = d

        self.potentials_q = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d // 2 + 1, self.H, dtype=dtype),
                    self.act,
                    nn.Linear(self.H, self.H, dtype=dtype),
                    self.act,
                    nn.Linear(self.H, 1, dtype=dtype),
                )
                for _ in range(self.nlayers)
            ]
        )

        self.potentials_pi = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d // 2 + 1, self.H, dtype=dtype),
                    self.act,
                    nn.Linear(self.H, self.H, dtype=dtype),
                    self.act,
                    nn.Linear(self.H, 1, dtype=dtype),
                )
                for _ in range(self.nlayers)
            ])
            
        if act_name=="sin":
            for i in range(self.nlayers):
                self.init(self.potentials_pi[i][0])
                self.init(self.potentials_pi[i][2])
                self.init(self.potentials_pi[i][4])
                
                self.init(self.potentials_q[i][0])
                self.init(self.potentials_q[i][2])
                self.init(self.potentials_q[i][4])
                
                
        ### adding single extra parameter to help with Hamiltonain matching
        self.h_0 = nn.Parameter(torch.zeros(1,dtype=dtype))
    
    # Project over the physical limit (just for non-conservative systems)
    def project_PL(self,sol):
        
        q1 = sol[:,:self.d//4]
        q2 = sol[:,self.d//4:2*self.d//4]
        pi1= sol[:,2*self.d//4:3*self.d//4]
        pi2= sol[:,3*self.d//4:]

        qsps = torch.cat(((q1+q2)/2,(q1+q2)/2,(pi1-pi2)/2,-(pi1-pi2)/2),dim=1)  
        return qsps      
    
    # Function to enforce the initial condition, essential that it vanishes at t=0       
    def f(self, t, i):
        return  torch.tanh(t)

    # Weight initialisation of weights when sin activation is used
    def init(self, layer, is_first=False):
            #Like in https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
            c = 6
            w0 = 30
            dim = self.d//2 if is_first else self.H
            w_std = (1 / dim) if is_first else (np.sqrt(c / dim) / w0)
            nn.init.uniform_(layer.weight,-w_std, w_std)
            nn.init.uniform_(layer.bias,-w_std, w_std)
    
    # Extracting the Hamiltonian behind SympFlow via automatic differentiation
    def Hamiltonian(self, z, t):
        q,pi = z[:,0:self.d//2], z[:,self.d//2:self.d]

        total_ham = 0
        for j in range(self.nlayers):
            i = self.nlayers - 1 - j
            
            pot_pi = lambda q,pi,t : self.potentials_pi[i](torch.cat((pi.reshape(-1,self.d//2),t.reshape(-1,1)),dim=1))
            pot_q = lambda q,pi,t : self.potentials_q[i](torch.cat((q.reshape(-1,self.d//2),t.reshape(-1,1)),dim=1))
            
            zz = torch.zeros_like(t)
            flow_q_inverse = lambda q,pi,t: q - (vmap(jacrev(pot_pi,argnums=1))(q,pi,t)-vmap(jacrev(pot_pi,argnums=1))(q,pi,zz)).reshape(-1,self.d//2)
            flow_pi_inverse = lambda q,pi,t: pi + (vmap(jacrev(pot_q,argnums=0))(q,pi,t)-vmap(jacrev(pot_q,argnums=0))(q,pi,zz)).reshape(-1,self.d//2)
            
            ham_pi = lambda q,pi,t: vmap(jacrev(pot_pi,argnums=2))(q,pi,t).reshape(-1,1)                 
            ham_q = lambda q,pi,t: vmap(jacrev(pot_q,argnums=2))(q,pi,t).reshape(-1,1)                       

            total_ham += ham_pi(q,pi,t)
            q = flow_q_inverse(q,pi,t)
            total_ham += ham_q(q,pi,t)
            pi = flow_pi_inverse(q,pi,t)
            
                          
        return total_ham + torch.tanh(self.h_0)
        
    
    # Forward pass implementing the SympFlow layers
    def forward(self,z,t):
        z = z.reshape(-1,self.d)
        t = t.reshape(-1,1)
        
        q,pi = z[:,:self.d//2], z[:,self.d//2:]

        for i in range(self.nlayers):
            pot_pi = lambda pi,t : self.potentials_pi[i](torch.cat((pi.reshape(1,self.d//2),t.reshape(1,1)),dim=1))
            pot_q = lambda q,t : self.potentials_q[i](torch.cat((q.reshape(1,self.d//2),t.reshape(1,1)),dim=1))
    
            zz = torch.zeros_like(t)
    
            pi = pi - (vmap(jacrev(pot_q,argnums=0))(q,t).reshape(-1,self.d//2)-vmap(jacrev(pot_q,argnums=0))(q,zz).reshape(-1,self.d//2)).reshape(-1,self.d//2)
            q = q + (vmap(jacrev(pot_pi,argnums=0))(pi,t).reshape(-1,self.d//2)-vmap(jacrev(pot_pi,argnums=0))(pi,zz).reshape(-1,self.d//2)).reshape(-1,self.d//2)
            
        output= torch.cat((q,pi),dim=1)
        if self.vec.isa_doubled_variables_system:
            output= self.project_PL(output)
        return output

#Architecture of a PINN to compare with SympFlow
class genericNet(nn.Module):
        def __init__(self,model_parameters = dict(
            hidden_nodes = 100,
            act_name = 'sin',
            nlayers = 3,
            dtype = torch.float,
            d = 4), vec = None, dt = 1.):
            super().__init__()
            
            ##The outputs of this network are to be intended as the variables
            ##q1,q2,pi1,pi2, i.e. the physical configuration variables together
            ##with the non-conservative momenta.
            
            neurons, act_name, nlayers, dtype, d = model_parameters['hidden_nodes'], model_parameters['act_name'], model_parameters['nlayers'], model_parameters['dtype'], model_parameters['d']
            # Assert that network dimension is compatible with vec dimension
            assert d == vec.ndim_total
            
            self.vec = vec 
            
            torch.manual_seed(1)
            np.random.seed(1)
            
            self.dtype = dtype
            self.vec = vec
            self.dt = dt
            
            activations = {
                "tanh": nn.Tanh(),
                "sin": sinAct(),
                "sigmoid": nn.Sigmoid(),
                "swish": swishAct(),
                "intTanh":integralTanh()
            }
            
            self.act = activations[act_name]
            
            self.H = neurons
            #### The dimensions here referes explicitly to the dimensions as seen by the network, i.e. the dimension of variable q
            #### So, if ndim_spatial = d and it is not double variables, then self.d = d, if it is double variables, then self.d = 2*d 
            self.d = d
            self.nlayers = nlayers
            
            self.firstLinear = nn.Linear(self.d+1,self.H,dtype=dtype)
            self.lastLinear = nn.Linear(self.H,self.d,dtype=dtype)
            if self.nlayers>2:
                self.linears = nn.ModuleList([nn.Linear(self.H,self.H,dtype=dtype) for _ in range(self.nlayers-2)])
                
            self.f = lambda t: torch.tanh(t) #it is just important that this vanishes at zero
        
        # Project over the physical limit (just for non-conservative systems)
        def project_PL(self,sol):
            
            q1 = sol[:,:self.d//4]
            q2 = sol[:,self.d//4:2*self.d//4]
            pi1= sol[:,2*self.d//4:3*self.d//4]
            pi2= sol[:,3*self.d//4:]

            qsps = torch.cat(((q1+q2)/2,(q1+q2)/2,(pi1-pi2)/2,-(pi1-pi2)/2),dim=1)  
            return qsps      

        # Forward pass specifying the network layers           
        def forward(self,z,t):
            
            z = z.reshape(-1,self.d)
            t = t.reshape(-1,1)
            res = self.act(self.firstLinear(torch.cat((z,t),dim=1)))
            if self.nlayers>2:
                for i in range(self.nlayers-2):
                    res = self.act(self.linears[i](res))
            res = self.lastLinear(res)
            res = z + self.f(t) * res
                
            if self.vec.isa_doubled_variables_system:
              res= self.project_PL(res)
      
            return res
        
