import torch
import torch.nn as nn
import torch.optim as optim


# 定义你的模型结构，例如一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    # 检查是否有可用的CUDA设备


if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA 设备对象
else:
    device = torch.device("cpu")  # a CPU 设备对象

# 初始化模型和优化器
model = SimpleNet(input_size=100, hidden_size=500, num_classes=10).to(device)  # 将模型移动到CUDA设备上
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 将优化器移动到CUDA设备上
criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到CUDA设备上

# 加载数据集（这里只是一个示例，请替换为您的数据集）
data = torch.randn(64, 100).to(device)  # 将数据移动到CUDA设备上
target = torch.randint(0, 10, (64,)).to(device)  # 将标签移动到CUDA设备上

# 训练循环
for epoch in range(100):  # 迭代100轮数据
    optimizer.zero_grad()  # 清空梯度
    output = model(data)  # 前向传播
    loss = criterion(output, target)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重