def load_uav_config(cfg):
    """Load UAV parameters from the selected simulation scenario."""
    print(f"Loading UAV scenario: {cfg.uav_scenarios.uav.case}")
    U = cfg.uav_scenarios.uav.U
    Power = cfg.uav_scenarios.uav.Power
    UAV_devs_dict = {key: value for key, value in cfg.uav_scenarios.uav.devices.items()}
    Joint_U_dict = {key: value for key, value in cfg.uav_scenarios.uav.joint_probabilities.items()}

    print(f"Loaded {len(U)} UAVs with device mappings.")
    return Power, U, UAV_devs_dict, Joint_U_dict
