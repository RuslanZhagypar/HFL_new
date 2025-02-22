import hydra
import numpy as np
import copy
import torch
from omegaconf import DictConfig
from .data_loader import load_dataset
from .model_loader import load_model
from utils.federated_utils import initialize_federated_learning
from .uav_loader import load_uav_config
from .local_update import LocalUpdate 
from utils.testing import test_img
import wandb
import random


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    wandb.init(project="NEW_PROJECT_HFL")

    # Set random seed for Python's built-in random module
    random.seed(cfg.seed)
    # Set random seed for NumPy
    np.random.seed(cfg.seed)
    # Set random seed for PyTorch
    torch.manual_seed(cfg.seed)


    for iterAVG in range(cfg.noOfExperiments):
        print(f"Loading dataset: {cfg.dataset.name}")

        # Load dataset and users
        dataset_train, dataset_test, dict_users, img_size = load_dataset(cfg)

        print(f"Dataset {cfg.dataset.name} loaded with {cfg.federation.num_users} users")
        print(f"Image size: {img_size}")

        # Load model
        net_glob = load_model(cfg, img_size)
        print(f"Model {cfg.model.name} loaded.")

        # Load BS-UAV-device mappings for selected case
        Power, U, UAV_devs_dict, Joint_U_dict = load_uav_config(cfg)
        Power = np.array(Power)
        U = np.array(U)

        # Initialize federated learning
        loss_train, acc_train, test_acc_train, test_loss_train, lweight, gweight = initialize_federated_learning(cfg, dict_users, UAV_devs_dict)

        # Group clients based on the federated learning configuration
        num_samples = np.array([len(dict_users[i]) for i in dict_users])
        num_clients = int(cfg.federation.num_users / cfg.federation.num_groups)
        group = []

        # Grouping clients into num_groups
        for i in range(0, cfg.federation.num_groups):
            if i == cfg.federation.num_groups - 1:
                group.append(np.arange(i * num_clients, cfg.federation.num_users))
            else:
                group.append(np.arange(i * num_clients, (i + 1) * num_clients))

        # Set group epochs and local iterations
        group_epochs = np.ones(cfg.federation.num_groups, dtype=int) * cfg.federation.group_freq
        local_iters = np.ones(cfg.federation.num_groups, dtype=int) * cfg.federation.local_period

        # Print the results
        print(f"Group epochs: {group_epochs[0]}")
        print(f"Local iterations: {local_iters[0]}")


    
        # Assuming you're inside the main training loop
        for iters in range(cfg.global_epochs):  # Use cfg.federation.epochs instead of args.epochs
            # Copy weights
            w_glob = net_glob.state_dict()
            net_glob_v2 = copy.deepcopy(net_glob).to(cfg.device)  # Use cfg.device for device
            w_glob_temp = net_glob_v2.state_dict()

            # Initialize weights to zero
            for k in w_glob_temp.keys():
                w_glob_temp[k] = 0

            ## Scheduling (Sampling)
            S = np.random.choice(cfg.federation.num_groups, size=(1, cfg.federation.num_groups), replace=False)  # Sampling (Scheduling)
            temp = np.random.rand(1, cfg.federation.num_groups)  # temp.shape = (1, M)
            # activeUAVs = [s for k, s in enumerate(S[0]) if temp[0, s] < cfg.federation.Power[s]]  # List of active UAVs based on power
            activeUAVs = [s for k, s in enumerate(S[0]) if temp[0, s] <  Power[s]]  # List of active UAVs based on power

            # Update UAVs based on additional success probability threshold
            S = np.random.choice(cfg.federation.num_groups, size=(1, cfg.federation.num_groups), replace=False)
            temp = np.random.rand(1, cfg.federation.num_groups)
            # activeUAVs = [s for k, s in enumerate(activeUAVs) if temp[0, s] < cfg.federation.U[s]]  # Refined active UAVs list
            activeUAVs = [s for k, s in enumerate(activeUAVs) if temp[0, s] < U[s]]  # Refined active UAVs list

            np.random.seed(60 + iters + iterAVG * 20)  # Set the random seed for this iteration

            for group_idx in activeUAVs:
                net_group = copy.deepcopy(net_glob).to(cfg.device)  # Use cfg.device
                net_group_v2 = copy.deepcopy(net_glob).to(cfg.device)

                loss_locals, acc_locals = [], []
                w_group = net_group.state_dict()
                w_group_local = net_group_v2.state_dict()

                for i in range(0, cfg.federation.group_freq):  # Use cfg.federation.group_freq for the number of group epochs
                    UAV_devs = UAV_devs_dict['U' + str(group_idx) + '_devs']  # List of devices in the chosen UAV cell
                    JointU = Joint_U_dict['Joint_U' + str(group_idx)]

                    # Sampling (Scheduling)
                    S_dev = UAV_devs  # All devices in the UAV cell
                    temp = np.random.rand(1, len(UAV_devs))  # temp.shape = (1, M)
                    activeDevs_per_UAV = [s for k, s in enumerate(S_dev) if temp[0, k] < JointU[k]]  # Participants' indices

                    net_group_local_v2 = copy.deepcopy(net_group).to(cfg.device)
                    w_local_previous = net_group_local_v2.state_dict()

                    net_group_local_v3 = copy.deepcopy(net_group).to(cfg.device)
                    w_local_temp = net_group_local_v3.state_dict()

                    # Initialize weights to zero
                    for k in w_local_temp.keys():
                        w_local_temp[k] = 0

                    for j in activeDevs_per_UAV:
                        idx = j
                        local = LocalUpdate(cfg=cfg, dataset=dataset_train, idxs=dict_users[idx],
                                            iters=cfg.federation.local_period, nums=num_samples[idx])  # Use cfg.optimization.local_period
                        net_group_local = copy.deepcopy(net_group).to(cfg.device)

                        w = local.train(net=net_group_local)

                        # Detach gradients to avoid memory overhead
                        for k in w.keys():
                            w[k] = w[k].detach()

                        if w_group is None:
                            # w_group = w
                            pass
                        else:
                            index_0 = UAV_devs.index(j)
                            index = UAV_devs.index(j)
                            for k in w.keys():
                                if cfg.algorithm == 'convHFL':
                                    w_local_temp[k] += (lweight[idx] / gweight[group_idx]) * (w[k] - w_local_previous[k])
                                elif cfg.algorithm == 'proposedHFL':
                                    w_local_temp[k] += (lweight[idx]/gweight[group_idx])/(JointU[index_0]) * (w[k] - w_local_previous[k])
                                else:
                                    print('Error: write a proper algorithm name')
                                # w_local_temp[k] += (lweight[idx]/gweight[group_idx])/(JointU[index_0]) * (w[k] - w_local_previous[k])
                                # w_group[k] += (lweight[index]/gweight[group_idx])/(JointU[index_0]) * (w[k] - w_local_previous[k])
                                # w_group[k] += (lweight[index]/gweight[group_idx])/(JointU[index_0]) * (w[k] - w_group[k])
                                # w_group[k] += (lweight[index]/gweight[group_idx]) * (w[k] - w_group[k])

                    # Update global weights for the group
                    for k in w_glob.keys():
                        w_group[k] += w_local_temp[k]
                    net_group.load_state_dict(w_group)

                for k in w_group.keys():
                    if cfg.algorithm == 'convHFL':
                        w_glob_temp[k] += gweight[group_idx] * (w_group[k] - w_group_local[k])
                    elif cfg.algorithm == 'proposedHFL':
                        # w_glob_temp[k] += gweight[group_idx]/(U[group_idx]) * (w_group[k] - w_group_local[k])
                        w_glob_temp[k] += gweight[group_idx]/(Power[group_idx]*U[group_idx]) * (w_group[k] - w_group_local[k])
                    else:
                        print('Error: write a proper algorithm name')

                    # w_glob[k] += gweight[group_idx] * (w_group[k] - w_glob[k])
                    

            # Aggregate the global weights after the group processing
            for k in w_glob.keys():
                w_glob[k] += w_glob_temp[k]

            # Copy the updated global weights to the global model
            net_glob.load_state_dict(w_glob)

            if (iters + 1) % cfg.federation.interval == 0:  # Use cfg.federation.interval
                with torch.no_grad():
                    # Test the model on training and testing datasets
                    acc_avg, loss_avg = test_img(copy.deepcopy(net_glob).to(cfg.device), dataset_train, cfg)
                    acc_test, loss_test = test_img(copy.deepcopy(net_glob).to(cfg.device), dataset_test, cfg)
                
                # Print the results for the current round
                print(f'Round {iters + 1:3d}, Training loss {loss_avg:.3f}', flush=True)
                print(f'Round {iters + 1:3d}, Training acc {acc_avg:.3f}', flush=True)
                print(f'Round {iters + 1:3d}, Test loss {loss_test:.3f}', flush=True)
                print(f'Round {iters + 1:3d}, Test acc {acc_test:.3f}', flush=True)
                wandb.log({'Accuracy': acc_test/100})
                wandb.log({'Loss': loss_test})



    
if __name__ == "__main__":
    main()