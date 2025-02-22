import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, cfg):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=cfg.optimization.test_bs, num_workers=cfg.system.num_workers)
    
    # device = torch.device(cfg.device)
    device = cfg.device
    
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)

    if cfg.system.verbose:
        print(f'\nTest set: Average loss: {test_loss:.4f} \nAccuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)\n')

    return accuracy.item(), test_loss
