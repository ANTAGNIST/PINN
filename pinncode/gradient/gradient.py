import torch
def gradient(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
            only_inputs=True
        )[0]
    else:
        return gradient(gradient(u, x), x, order=order-1)