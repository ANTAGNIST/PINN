import torch
class MLP(torch.nn.Module):
    def __init__(self, input_num, output_num):
        # input_num表示输入x的列数，output_num代表输出y的列数
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_num, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, output_num)
        )
    def forward(self, x):
        return self.net(x)