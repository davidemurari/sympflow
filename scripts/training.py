import torch
import numpy as np
from torch.func import vmap, jacfwd
import time as time_lib
from tqdm import tqdm
from scripts.sampling import sample_ic
import scripts.settings as settings

torch.set_printoptions(precision=10)

####
# This script contains a training loop for the unsupervised experiments, i.e., trainModel, and one for the supervised experiments, i.e., trainModelSupervised.
# The supervised training loss is much simpler.

def trainingSupervised(
        model,
        vec,
        num_epochs,
        trainloader,
        optimizer,
        criterion,
        scheduler=None,
        device='cpu',
        is_energy_reg=False
    ):
    
    for epoch in (pbar :=tqdm(range(num_epochs))):
        model.train()  # Set model to training mode
        for batch_idx, (x, t, y) in enumerate(trainloader):
            
            optimizer.zero_grad()
            x,t,y = x.to(device),t.to(device),y.to(device)
            predictions = model(x, t)
            loss = criterion(predictions, y)

            if hasattr(model, "Hamiltonian") and is_energy_reg:
                loss += torch.mean(
                    (model.Hamiltonian(predictions, t) - model.Hamiltonian(x, torch.zeros_like(t))) ** 2
                )
                
            loss.backward()
            optimizer.step()
            if not scheduler==None:
                scheduler.step()
        pbar.set_postfix_str(str(loss.item()))

    print("Training complete.\n")

def trainModel(
    model,
    training_parameters,
    vec,
    system_parameters,
    optimizer,
    test_set,
    ode_name
):
    # unpack training parameters
    dt, tf, t0, n_train, epochs, device, dtype = (
        training_parameters["dt"],
        training_parameters["tf"],
        training_parameters["t0"],
        training_parameters["n_train"],
        training_parameters["epochs"],
        training_parameters["device"],
        training_parameters["dtype"],
    )

    timestamp = time_lib.strftime("%Y%m%d_%H%M%S")

    name_txt_losses = f"{settings.paths['losses']}{ode_name}/{training_parameters['name_run']}/TrainingLosses_{timestamp}.txt"
    name_txt_test_losses = f"{settings.paths['losses']}{ode_name}/{training_parameters['name_run']}/TestLosses_{timestamp}.txt"
    
    Loss_history = []
    Residual_loss_test = []
    TeP0 = time_lib.time()

    best_loss = 100.0

    z_test, t_test = test_set["z"], test_set["t"]

    if not hasattr(model, "Hamiltonian") and (vec.isa_doubled_variables_system) and training_parameters["hamReg"]:
            print("\n\nThe Hamiltonian regularisation for MLPs with double variable systems is not implemented. The model will be trained without regularisation.\n\n")

    for epoch in (pbar := tqdm(range(epochs))):
        
        factor = 1.1 
        z, t = sample_ic(vec.system_parameters, vec, dtype, n_train, dt, factor, t0)
        z, t = z.to(device), t.to(device)

        loss = 0.0
        
        optimizer.zero_grad()

        derivative = lambda x, t: vmap(jacfwd(model, argnums=1))(x, t)
        z_d = derivative(z, t).squeeze()
        z_out = model(z, t)

        # PINN residual loss
        Lres = vec.residual_loss(z_out, z_d)

        fact = 2 if vec.isa_doubled_variables_system else 1 #to account for the correct dimensions

        loss += Lres
        
        #Computing the regularisation if required to do so (i.e. in hamReg and pinnReg)
        
        if not hasattr(model, "Hamiltonian") and (not vec.isa_doubled_variables_system) and training_parameters["hamReg"]:
            z_t = model(z, t)
            
            loss += torch.mean(
                (
                    vec.eval_hamiltonian(z[:,:fact*vec.ndim_spatial], z[:,fact*vec.ndim_spatial:])
                    - vec.eval_hamiltonian(z_t[:,:fact*vec.ndim_spatial], z_t[:,fact*vec.ndim_spatial:])
                )** 2
            )
            
        if hasattr(model, "Hamiltonian") and training_parameters["hamReg"]:
            L_hamiltonian_match = torch.mean(
                (model.Hamiltonian(z, t) - vec.eval_hamiltonian(z[:, :fact*vec.ndim_spatial], z[:, fact*vec.ndim_spatial:])) ** 2
            )
            loss += L_hamiltonian_match

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        pbar.set_postfix_str(str(loss.item()))

        # Check test set
        model.eval()
        with torch.no_grad():
            derivative = lambda x, t: vmap(jacfwd(model, argnums=1))(x, t)
            z_d = derivative(z_test, t_test).squeeze()
            z_out = model(z_test, t_test)
            residual_loss = vec.residual_loss(z_out, z_d)
            with open(name_txt_test_losses, "a") as myfile:
                myfile.write(str(residual_loss.item()) + "\n")
                Residual_loss_test.append(residual_loss.item())

            # Save the best model encountered in the last 15% of the epochs
            if epoch > int(0.85 * epochs):
                if residual_loss.item() < best_loss:
                    torch.save(
                        model.state_dict(),
                        settings.paths["model"]
                        + f"tmpFiles/{ode_name}_{training_parameters['name_run']}_bestModel.pt",
                    )
                    best_loss = residual_loss.item()

        model.train()

        with open(name_txt_losses, "a") as myfile:
            myfile.write(str(loss.item()) + "\n")
            Loss_history.append(loss.item())

    TePf = time_lib.time()
    runTime = TePf - TeP0

    # Load the best found model
    path_best_model = (
        settings.paths["model"]
        + f"tmpFiles/{ode_name}_{training_parameters['name_run']}_bestModel.pt"
    )
    model.load_state_dict(torch.load(path_best_model, map_location=device))

    print("Training complete.\n")
    return Loss_history