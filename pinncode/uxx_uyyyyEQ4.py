import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gradient.gradient import gradient
from Model.MLP import MLP
from RandomSeed.RandomSeed import setup_seed
from ConvertDataType.CpuAndGrad2Cuda import cpu_and_grad2cuda
from ConvertDataType.CpuAndGrad2ndarray import cpu_and_grad2ndarray
from LossFunction.LossRMSE import loss_fn

epoches = 10000
h = 100 #绘图网格密度
N = 1000 #内点配置点数
N1 = 100 #边界点配置点数
N2 = 100 #PDE数据点
# cuda cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 随机种子
setup_seed(8)
# 创建tensor_cpu_require_grad数组，数组命名规则：数组变量_变量eq数字Nx
data = {
    "xN1":torch.randn(N1, 1).requires_grad_(True),
    "yN1":torch.randn(N1, 1).requires_grad_(True),
    "xN":torch.randn(N, 1).requires_grad_(True),
    "yN":torch.randn(N, 1).requires_grad_(True),
    "xN2":torch.randn(N2, 1).requires_grad_(True),
    "yN2":torch.randn(N2, 1).requires_grad_(True),
     }
# 创建model函数:u
u_net = MLP(2, 1)
# 控制方程u(x,y) 随机N个(x,y)点
data["uN"] = u_net(torch.cat((data["xN"], data["yN"]), dim=1))
# 控制方程d2u(x,y)_dx2 随机N个(x,y)点
data["d2u(x,y)_dx2N"] = gradient(data["uN"], data["xN"], order=2)
# 控制方程d4u(x,y)_dx4 随机N个(x,y)点
data["d4u(x,y)_dx4N"] = gradient(data["uN"], data["xN"], order=4)
# u(x,0) 随机N1个(x,0)点
data["u(x,0)N1"] = u_net(torch.cat((data["xN1"], data["yN1"]*0), dim=1))
# u(x,1) 随机N1个(x,1)点
data["u(x,1)N1"] = u_net(torch.cat((data["xN1"], data["yN1"]*0+1), dim=1))
# u(0,y) 随机N1个(0,y)点
data["u(0,y)N1"] = u_net(torch.cat((data["xN1"]*0, data["yN1"]), dim=1))
# u(1,y) 随机N1给(1,y)点
data["u(1,y)N1"] = u_net(torch.cat((data["xN1"]*0+1, data["yN1"]), dim=1))
# d2u(x,0)_dy2 随机N1个(x,0)点
data["d2u(x,0)_dy2N1"] = gradient(data["u(0,y)N1"], data["yN1"], order=2)
# d2u(x,1)_dy2 随机N1个(x,1)点
data["d2u(x,1)_dy2N1"] = gradient(data["u(x,1)N1"], data["yN1"], order=2)
"""
导数创建套路：
1.先通过u_net(torch.cat((data["x"]或data["x"]*0或data["x"]*1, data["y"]或data["y"]*0或data["y"]*1), dim=1))创建位于(x=x或0或1, y=y或0或1)变量u
2.再通过gradient(u, data["x"], order=i)创建diu_dxi导数
"""
# 创建tensor_cuda数组
data_cuda = cpu_and_grad2cuda(data)
# 创建ndarray数组
data_ndarray = cpu_and_grad2ndarray(data)
# 根据论文，https://maziarraissi.github.io/PINNs/创建loss
# d2u_dx2 - d4u_dy4 = (2-x^2)*e^(-y)
loss_interior = loss_fn(data_cuda["d2u(x,y)_dx2N"] - data_cuda["d4u(x,y)_dx4N"] - (2-data_cuda["xN"]) * torch.exp(-data_cuda["yN"]))
# d2u(x,0)_dy2 = x^2
loss_down_yy = loss_fn(data_cuda["d2u(x,0)_dy2N1"], data_cuda["xN1"]**2)
# d2u(x,1)_dy2 = x^2/e
loss_up_yy = loss_fn(data_cuda["d2u(x,1)_dy2N1"], data_cuda["xN1"]**2/torch.e)
# u(x,0) = x^2
loss_down = loss_fn(data_cuda["u(x,0)N1"], data_cuda["xN1"])
# u(x,1) = x^2/e
loss_up = loss_fn(data_cuda["u(x,1)N1"], data_cuda["xN1"]**2/torch.e)
# u(0,y) = 0
loss_left = loss_fn(data_cuda["u(0,y)N1"], 0)
# u(1,y) = e^(-y)
loss_right = loss_fn(data_cuda["u(1,y)N1"], torch.exp(-data_cuda["yN1"]))
# 将loss_down_yy到loss_right的误差相加
loss = loss_interior + loss_down_yy + loss_up_yy + loss_down + loss_up + loss_left + loss_right
# 创建优化器
optimizer = torch.optim.Adam(u_net.parameters(), lr=0.001)

u_net.train()
u_net = u_net.to(device)
for epoch in range(epoches):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch={epoch},loss={loss}")