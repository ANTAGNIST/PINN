import torch

def cpu_and_grad2cuda(data):
    """
    :param data: 类型属于cpu,require_grad=True
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cuda = dict(data)
    for key, value in data_cuda.items():
        data_cuda[key] = value.clone().to(device)
    return data_cuda