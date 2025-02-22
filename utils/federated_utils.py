def initialize_federated_learning(cfg, dict_users, UAV_devs_dict):
    """Initialize training metrics and compute client/group weights."""
    print("Initializing Federated Learning...")

    # Tracking metrics
    loss_train = []
    acc_train = []
    test_acc_train = []
    test_loss_train = []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # Compute local weights for clients
    total_train_samples = 60000 if cfg.dataset.name == "mnist" else len(dict_users)  # Adjust for dataset
    lweight = [len(dict_users[k]) / total_train_samples for k in range(cfg.federation.num_users)]


    # Compute group weights for UAVs
    gweight = []
    for u in range(cfg.federation.num_groups):
        p_per_uav = sum(lweight[k] for k in UAV_devs_dict['U' + str(u) + '_devs'])
        gweight.append(p_per_uav)

    print("Initialization complete.")
    return loss_train, acc_train, test_acc_train, test_loss_train, lweight, gweight
