# **UAV-assisted Unbiased Hierarchical Federated Learning: Performance and Convergence Analysis**  
Implementation of the code from the paper "[UAV-assisted Unbiased Hierarchical Federated Learning: Performance and Convergence Analysis"  
[Link]  

## **Description**
The purpose of this code is to reproduce the results and experiments described in the paper. The dataset is MNIST, and the model is MLP. However, these can be changed if needed. The repository contains python files for model training and Matlab and Mathematica files for stochastic geometry calculations. The project uses _Hydra_ package for the ease of conducting experiements. 

## **Getting Started**
To run the training, a user can look up the _.configs_ files to check the parameter values. Then, a user can execute: 
_python -m src.main_ 
The parameter values can be changed directly in the _configs.py_ file, or on the command line, e.g.:
_python -m src.main algorithm=new_algorithm_



### **Prerequisites**
Before running the code, ensure that you have the following dependencies installed:
- **Python** (version 3.x or higher)
- **PyTorch** (or other frameworks used in the paper, e.g., TensorFlow, etc.)
- Other dependencies listed below

You can install the necessary libraries using `pip`:

```bash
pip install -r requirements.txt
