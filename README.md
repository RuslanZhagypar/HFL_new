# UAV-assisted Unbiased Hierarchical Federated Learning: Performance and Convergence Analysis

Implementation of the code from the paper "[UAV-assisted Unbiased Hierarchical Federated Learning: Performance and Convergence Analysis](Link)"

## Description
The purpose of this code is to reproduce the results and experiments described in the paper. The dataset is MNIST, and the model is MLP. However, these can be changed if needed. The repository contains Python files for model training and MATLAB and Mathematica files for stochastic geometry calculations. The project uses the **Hydra** package for the ease of conducting experiments.

## Getting Started
To run the training, a user can look up the *configs* files to check the parameter values. Then, a user can execute:

```bash
python -m src.main


The parameter values can be changed directly in the `_config.yaml_` file, or on the command line, e.g.:
```bash
python -m src.main algorithm=convHFL uav_scenarios=case2


A user can use **Mathematica** and/or **MATLAB** files to generate new probabilities for different system parameter values. The obtained probability values are then used to create `case_2.yaml`, `case_3.yaml`, etc., configuration files. Then, different cases can be chosen in the `_config.yaml_` file.

## Prerequisites
Before running the code, ensure that you have the following dependencies installed:

- **Python** (version 3.10 or higher)
- **PyTorch** (version 2.4.1)
- **CUDA** (version 12.4)
- Other dependencies listed in the `requirements.txt` file

You can install the necessary libraries using `pip`:


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

