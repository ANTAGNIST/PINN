import torch

def loss_fn(pred, true=None):
    # pred与true输入顺序变换不影响结果
    if true == None:
        return torch.mean(pred**2)
    else:
        return torch.mean((pred - true) ** 2)