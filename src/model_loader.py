import torch
import torch.nn as nn
from model.networks import MLP, CNNMnist

def load_model(cfg, img_size):
    """Load the model based on Hydra configuration."""
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = cfg.device

    if cfg.model.name == "cnn":
        model = CNNMnist(args=cfg).to(device)
    elif cfg.model.name == "mlp":
        len_in = 1
        for x in img_size:
            len_in *= x
        model = MLP(dim_in=len_in, dim_hidden=cfg.model.hidden_dim, dim_out=cfg.model.num_classes).to(device)
        model = nn.DataParallel(model)
    else:
        raise ValueError("Error: Unrecognized model type")
    model.train()
    return model
