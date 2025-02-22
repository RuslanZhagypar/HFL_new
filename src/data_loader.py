import torch
from torchvision import datasets, transforms

from utils.sampling import mnist_iid, mnist_noniid

def load_dataset(cfg):
    """Load dataset and split users based on Hydra configuration."""
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if cfg.dataset.name == "mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(cfg.dataset.data_dir, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(cfg.dataset.data_dir, train=False, download=True, transform=trans_mnist)
        
        # Sample users
        if cfg.dataset.iid:
            dict_users = mnist_iid(dataset_train, cfg.federation.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, cfg.federation.num_users)
    else:
        raise ValueError("Error: Unrecognized dataset")

    img_size = dataset_train[0][0].shape
    return dataset_train, dataset_test, dict_users, img_size

    


