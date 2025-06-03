from abc import abstractmethod, ABC
from math import sqrt


import torch

dtype=torch.float32

"Here we harcode the physical parameters for the different toy problems"


###vec_field_name corrseponds to the name of the class in vector_field, corresponding to the system

SimpleHO_exp = dict(
    system="harmonic-oscillator",
    m=1.0,
    k=1.0,
    ll=0.0,
    qub=1.2,
    qlb=-1.2,
    piub=1.2,
    pilb=-1.2,
    ndim_spatial = 1,
    q0=torch.tensor([1.],dtype=dtype),
    pi0=torch.tensor([0.],dtype=dtype),
    vec_field_name = "HarmonicOscillator"
)

DampedHO_exp = dict(
    system="damped-harmonic-oscillator",
    m = 1.0,
    k = 1,
    ll = 0.1,
    qub= 1.2,
    qlb= -1.2,
    piub= 1.2,
    pilb= -1.2,
    ndim_spatial = 1,
    q0=torch.tensor([1.,1.],dtype=dtype),
    pi0=torch.tensor([0.0,0.0],dtype=dtype),
    vec_field_name = "DampedHarmonicOscillator"
)

Henon_Heiles_exp = dict(
    system="henon-heiles",
    qub= 1.0,
    qlb=-1.0,
    piub=1.0,
    pilb=-1.0,
    l = 1.,
    ndim_spatial = 2,
    q0 = torch.tensor([0.3,-0.3],dtype=dtype),
    pi0 = torch.tensor([0.3,0.15],dtype=dtype),
    vec_field_name = "HenonHeiles"
)