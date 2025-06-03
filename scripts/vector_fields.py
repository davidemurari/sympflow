from abc import abstractmethod, ABC
from math import sqrt

import torch
import os
import sys
import numpy as np

def bnorm(v):
    #if a numpy array then return the numpy norm 
    if type(v) == np.ndarray: 
        if(len(v.shape)==1): return np.linalg.norm(v,axis=0,keepdims=True,ord=2)
        return np.linalg.norm(v,axis=1,keepdims=True,ord=2)
    #Computes the norm for a batch of inputs, if batch exists, otherwise normal norm
    if(len(v.shape)==1): return torch.linalg.norm(v,dim=0,keepdims=True,ord=2)
    return torch.linalg.norm(v,dim=1,keepdims=True,ord=2)

def sum_1(v):
    #if a numpy array then return the numpy norm 
    if type(v) == np.ndarray: 
        if(len(v.shape)==1): return np.sum(v,keepdims=True)
        return np.sum(v,axis=1,keepdims=True)
    #Computes the norm for a batch of inputs, if batch exists, otherwise normal norm
    if(len(v.shape)==1): return torch.sum(v,dim=0,keepdims=True)
    return torch.sum(v,dim=1,keepdims=True)


class HamiltonianVecField(ABC):
    """Abstract class defining the interface for a Hamiltonian vector field"""
    
    def __init__(self, system_parameters):
        self._system_parameters = system_parameters
        self._ndim_spatial = system_parameters['ndim_spatial']
    
    @abstractmethod
    def __repr__(self):
        pass

    @property
    @abstractmethod
    def isa_doubled_variables_system(self):
        pass
    
    @property
    def ndim_spatial(self):
        return self._ndim_spatial

    @property
    def system_parameters(self):
        return self._system_parameters

    @property
    def ndim_total(self):
        if self.isa_doubled_variables_system:
            return self.ndim_spatial * 4
        else:
            return self.ndim_spatial * 2

    @abstractmethod
    def eval_vec_field(self, x, pi):
        pass

    @abstractmethod
    def eval_hamiltonian(self, x, pi):
        pass
    
    def eval_energy(self, x, pi):
        if not self.isa_doubled_variables_system: return self.eval_hamiltonian(x, pi)

    def residual_loss(self, z, z_ad):
        x,pi = z[:,:self.ndim_total//2],z[:,self.ndim_total//2:]
        x_ad,pi_ad = z_ad[:,:self.ndim_total//2],z_ad[:,self.ndim_total//2:]
        # torch.mean will fail if x.shape != x_ad.shape
        # or if pi.shape != pi_ad.shape, but can not see if
        # x.shape != pi.shape
        assert x.shape == pi.shape
        x_vecfield, pi_vecfield = self.eval_vec_field(x, pi)
        return 0.5 * (
            torch.mean((x_vecfield - x_ad) ** 2)
            + torch.mean((pi_vecfield - pi_ad) ** 2)
        )

    def true_solution(self, t, x_0, pi_0):
        raise NotImplementedError
        
class HarmonicOscillator(HamiltonianVecField):
    # def __init__(self, k=1.0, m=1.0, ndim_spatial=1):
    def __init__(self, system_parameters):
        super().__init__(system_parameters)
        
        self.k = system_parameters['k']
        self.m = system_parameters['m']

    def __repr__(self):
        return f"{self.ndim_spatial}D-Harmonic oscillator with k={self.k:.1e}, m={self.m:.1e}"

    @property
    def isa_doubled_variables_system(self):
        return False

    def eval_vec_field(self, x, pi):
        assert x.shape[-1]==self.ndim_spatial
        assert pi.shape[-1]==self.ndim_spatial
        
        return pi / self.m, -self.k * x

    def eval_hamiltonian(self, x, pi):
        assert x.shape[-1]==self.ndim_spatial
        assert pi.shape[-1]==self.ndim_spatial
        
        x_norm = bnorm(x)
        pi_norm = bnorm(pi)
        return 0.5 * (self.k * x_norm**2 + pi_norm**2 / self.m)

    def true_solution(self, ts, x_0, pi_0):
        omega = sqrt(self.k / self.m)
        return (
            x_0 * torch.cos(omega * ts)
            + (pi_0 / (self.m * omega)) * torch.sin(omega * ts),
            -x_0 * self.m * omega * torch.sin(omega * ts)
            + pi_0 * torch.cos(omega * ts),
        )
        

class HenonHeiles(HamiltonianVecField):
    def __init__(self, system_parameters):
        super().__init__(system_parameters)
        
        self.l = system_parameters['l'] #lambda parameter

    def getx1x2(self,x):
        if len(x.shape)==2:
            return x[:,0:1],x[:,1:2]
        else:
            return x[0],x[1]

    def __repr__(self):
        return f"{self.ndim_spatial}D-Henon Heiles with lambda={self.l:.1e}"

    @property
    def isa_doubled_variables_system(self):
        return False

    def eval_vec_field(self, x, pi):
        
        assert x.shape[-1]==self.ndim_spatial
        assert pi.shape[-1]==self.ndim_spatial
        
        x1,x2 = self.getx1x2(x)
        if type(x) == np.ndarray:
            if len(x.shape)==1:
                return pi, np.array([-x1-2*self.l*x1*x2,-x2-self.l*(x1**2-x2**2)])
            else:
                return pi, np.concatenate((-x1-2*self.l*x1*x2,-x2-self.l*(x1**2-x2**2)),axis=1)
        else:
            if len(x.shape)==1:
                return pi, torch.tensor([-x1-2*self.l*x1*x2,-x2-self.l*(x1**2-x2**2)])
            else:    
                return pi, torch.cat([-x1-2*self.l*x1*x2,-x2-self.l*(x1**2-x2**2)],dim=1)

    def eval_hamiltonian(self, x, pi):
        
        assert x.shape[-1]==self.ndim_spatial
        assert pi.shape[-1]==self.ndim_spatial
        
        x_norm = bnorm(x)
        pi_norm = bnorm(pi)
        x1,x2 = self.getx1x2(x)
        return 0.5 * (x_norm**2 + pi_norm**2) + self.l * (x1**2*x2-x2**3/3)

    def true_solution(self, ts, x_0, pi_0):
        
        omega = sqrt(self.k / self.m)
        return (
            x_0 * torch.cos(omega * ts)
            + (pi_0 / (self.m * omega)) * torch.sin(omega * ts),
            -x_0 * self.m * omega * torch.sin(omega * ts)
            + pi_0 * torch.cos(omega * ts),
        )


class DampedHarmonicOscillator(HamiltonianVecField):
    def __init__(self, system_parameters):
        super().__init__(system_parameters)
        
        self.k = system_parameters['k']
        self.m = system_parameters['m']
        self.ll = system_parameters['ll']
        
    def __repr__(self):
        return f"{self.ndim_spatial}D-Damped harmonic oscillator with k={self.k:.1e}, m={self.m:.1e}, lambda={self.ll:.1e}"

    @property
    def isa_doubled_variables_system(self):
        return True

    def physical_energy(self, x, pi):
        #x, pi should be 1d, so the physical variables
        return pi**2 / (2*self.m) + self.k/2 * x**2

    def eval_vec_field(self, x, pi):
        
        assert x.shape[-1]==2*self.ndim_spatial
        assert pi.shape[-1]==2*self.ndim_spatial
        
        x1,x2 = x[:,:self.ndim_total//4],x[:,self.ndim_total//4:]
        pix1,pix2 = pi[:,:self.ndim_total//4],pi[:,self.ndim_total//4:]
        
        x_deriv = torch.cat((pix1/(self.m) + self.ll/(2*self.m)*(x1-x2), -pix2/self.m - self.ll/(2*self.m)*(x1-x2)),dim = 1)
        pi_deriv = torch.cat((-self.ll/(2*self.m)*(pix1-pix2) - self.k*x1, self.ll/(2*self.m)*(pix1-pix2) + self.k*x2),dim = 1)
        return x_deriv, pi_deriv

    def eval_hamiltonian(self, x, pi):
        
        assert x.shape[-1]==2*self.ndim_spatial
        assert pi.shape[-1]==2*self.ndim_spatial
        x1,x2 = x[:,:self.ndim_total//4],x[:,self.ndim_total//4:]
        pix1,pix2 = pi[:,:self.ndim_total//4],pi[:,self.ndim_total//4:]
        x1_norm = bnorm(x1)
        x2_norm = bnorm(x2)
        pi1_norm = bnorm(pix1)
        pi2_norm = bnorm(pix2)

        return (pi1_norm**2-pi2_norm**2)/(2*self.m) + self.ll/(2*self.m) * sum_1((x1-x2)*(pix1-pix2)) + (self.k/2)*(x1_norm**2-x2_norm**2)
    
    def eval_energy(self, x, pi):
        assert x.shape[-1]==2*self.ndim_spatial
        assert pi.shape[-1]==2*self.ndim_spatial
        
        ## Note, that here the energy is not the Hamiltonian, but energy of the particle, so x and pi are half the dimension of the full system
        x_norm = bnorm(x)
        pi_norm = bnorm(pi)
        return 0.5 * (self.k * x_norm**2 + pi_norm**2 / self.m)

if __name__ == "__main__":
    ho = HarmonicOscillator(k=10, m=1, ndim_spatial=1)
    import matplotlib.pyplot as plt

    ts = torch.linspace(0, 10, 500)
    x_0, pi_0 = torch.tensor(-1.0), torch.tensor(1.0)
    xs, pis = ho.true_solution(ts, x_0, pi_0)
    x_dots, pi_dots = torch.vmap(
        torch.func.jacfwd(ho.true_solution, argnums=0), in_dims=(0, None, None)
    )(ts, x_0, pi_0)

    h = 1e-5
    x_hs, pi_hs = ho.true_solution(ts + h, x_0, pi_0)

    x_hdots = (x_hs - xs) / h
    pi_hdots = (pi_hs - pis) / h
    plt.subplot(2, 2, 1)
    plt.plot(ts, xs)
    plt.plot(ts, pi_dots)
    plt.legend(["$x$", "$\\dot\\pi$"])
    plt.subplot(2, 2, 2)
    plt.plot(ts, pis)
    plt.plot(ts, x_dots)
    plt.legend(["$\\pi$", "$\\dot x$"])
    plt.subplot(2, 2, 3)
    plt.plot(ts, xs)
    plt.plot(ts, x_dots)
    plt.plot(ts, x_hdots)
    plt.legend(["$x$", "$\\dot x$", "$\\dot x_h$"])
    plt.subplot(2, 2, 4)
    plt.plot(ts, pis)
    plt.plot(ts, pi_dots)
    plt.plot(ts, pi_hdots)
    plt.legend(["$\\pi$", "$\\dot \\pi$", "$\\dot\\pi_h$"])
    plt.show()
