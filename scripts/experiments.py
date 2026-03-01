from copy import deepcopy


import torch

dtype=torch.float32

"Here we harcode the physical parameters for the different toy problems"


###vec_field_name corresponds to the name of the class in vector_field, corresponding to the system

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
    ll = 0.01,
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


def get_system_parameters(ode_name: str, ll: float | None = None) -> dict:
    """Return a copy of system parameters with optional damping override.

    Inputs:
    - ode_name: one of {"SimpleHO", "DampedHO", "HenonHeiles"}.
    - ll: optional damping coefficient override (used only for DampedHO).

    Returns:
    - Dictionary of system parameters.
    """
    if ode_name == "SimpleHO":
        return deepcopy(SimpleHO_exp)
    if ode_name == "HenonHeiles":
        return deepcopy(Henon_Heiles_exp)
    if ode_name == "DampedHO":
        params = deepcopy(DampedHO_exp)
        if ll is not None:
            params["ll"] = float(ll)
        return params
    raise ValueError(f"Unsupported ode_name '{ode_name}'. Expected one of: SimpleHO, DampedHO, HenonHeiles.")
