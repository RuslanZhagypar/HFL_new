# **UAV-assisted Unbiased Hierarchical Federated Learning: Performance and Convergence Analysis**  
Implementation of the code from the paper "[UAV-assisted Unbiased Hierarchical Federated Learning: Performance and Convergence Analysis"  
[Link]  

## **Description**
The purpose of this code is to reproduce the results and experiments described in the paper. The dataset is MNIST, and the model is MLP. However, these can be changed if needed. The repository contains python files for model training and Matlab and Mathematica files for stochastic geometry calculations. The project uses _Hydra_ package for the ease of conducting experiements. 

## **Getting Started**
To run the training, a user can look up the *configs* files to check the parameter values. Then, a user can execute: 

```
python -m src.main
```

The parameter values can be changed directly in the _config.yaml_ file, or on the command line, e.g.:

```
_python -m src.main algorithm=convHFL uav_scenarios=case2_
```

A user can use Mathematica or/and Matlab files to generate new probabilities for a different system parameter values. The obtained probability values are then used to create _case_2.yaml, case_3.yaml,..._ configuration files. Then different cases can be chosen in the _config.yaml_ file.





### **Prerequisites**
Before running the code, ensure that you have the following dependencies installed:
- **Python** (version Python 3.10 or higher)
- **PyTorch=2.4.1**
- **CUDA version 12.4** 
- Other dependencies listed in the requirements.txt file

You can install the necessary libraries using `pip`:

```bash
pip install -r requirements.txt

format it properly for github readme file
