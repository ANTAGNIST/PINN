import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gradient.gradient import gradient
from Model.MLP import MLP
from RandomSeed.RandomSeed import setup_seed

epoches = 10000
h = 100 #绘图网格密度
N = 1000 #内点配置点数
N1 = 100 #边界点配置点数
N2 = 100 #PDE数据点
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
uN = u_net(torch.cat((data["xN"], data["yN"]), dim=1))
# 控制方程d2u(x,y)_dx2 随机N个(x,y)点
d2u_dx2N = gradient(uN, data["xN"], order=2)
# 控制方程d4u(x,y)_dx4 随机N个(x,y)点
d4u_dy4N = gradient(uN, data["xN"], order=4)
# u(x,0) 随机N1个(x,0)点
u_yeq0N1 = u_net(torch.cat((data["xN1"], data["yN1"]*0), dim=1))
# u(x,1) 随机N1个(x,1)点
u_yeq1N1 = u_net(torch.cat((data["xN1"], data["yN1"]*1), dim=1))
# u(0,y) 随机N1个(0,y)点
u_xeq0N1 = u_net(torch.cat((data["xN1"]*0, data["yN1"]), dim=1))
# u(1,y) 随机N1给(1,y)点
u_xeq1N1 = u_net(torch.cat((data["xN1"]*1, data["yN1"]), dim=1))
# d2u(x,0)_dy2 随机N1个(x,0)点
d2u_yeq0_dx2N1 = gradient(u_xeq0N1, data["yN1"], order=2)
# d2u(x,1)_dy2 随机N1个(x,1)点
d2u_yeq1_dy2N1 = gradient(u_yeq1N1, data["yN1"], order=2)
"""
导数创建套路：
1.先通过u_net(torch.cat((data["x"]或data["x"]*0或data["x"]*1, data["y"]或data["y"]*0或data["y"]*1), dim=1))创建位于(x=x或0或1, y=y或0或1)变量u
2.再通过gradient(u, data["x"], order=i)创建diu_dxi导数
"""
# 将变量输入进data中
data["d2u(x,y)_dx2N"], data["d4u(x,y)_dx4N"], data["u(x,0)N1"], data["u(x,1)N1"] = d2u_dx2N, d4u_dy4N, u_yeq0N1, u_yeq1N1
data["u(0,y)N1"], data["u(1,y)N1"], data["d2u(x,0)_dy2N1"], data["d2u(x,1)_dy2N1"] = u_xeq0N1, u_xeq1N1, d2u_yeq0_dx2N1, d2u_yeq1_dy2N1
# 创建tensor_cuda数组
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_cuda = dict(data)
for key, value in data_cuda.items():
    data_cuda[key] = value.clone().to(device)
# 创建ndarray数组
data_ndarray = dict(data)
for key, value in data_ndarray.items():
    data_ndarray[key] = value.detach().cpu().numpy()

# 创建损失函数
def loss_fn(pred, true=None):
    # pred与true输入顺序变换不影响结果
    if true == None:
        return torch.mean(pred**2)
    else:
        return torch.mean((pred - true) ** 2)
# 根据论文，https://maziarraissi.github.io/PINNs/
# loss_interior直接用mse表示；其他用了u_net的结果与真实结果必然不同，然后使用了u_net()的左式与右式先用mse，然后再相加mse
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
loss_all = loss_down_yy + loss_up_yy + loss_down + loss_up + loss_left + loss_right
# 尝试找到是否是要先将u_net放到cuda中，才能让Adam中的device_grads也在cuda中
u_net.to(device)
# # 创建优化器
optimizer = torch.optim.Adam(u_net.parameters(), lr=0.001)
# optimizer2 = torch.optim.LBFGS(
#         u_net.parameters(),
#         lr=1.0,
#         max_iter=50000,
#         max_eval=50000,
#         history_size=50,
#         tolerance_grad=1e-5,
#         tolerance_change=1.0 * np.finfo(float).eps,
#         line_search_fn="strong_wolfe"
# )
# optimizer2需要传递一个闭包参数才能运行
# https://blog.csdn.net/awslyyds/article/details/127503167
# class PINN():
#     def __init__(self, epoches, model, *loss):
#         # 创建优化器
#         self.optimizer = torch.optim.Adam(u_net.parameters(), lr=0.001)
#         self.optimizer2 = torch.optim.LBFGS(
#             u_net.parameters(),
#             lr=1.0,
#             max_iter=50000,
#             max_eval=50000,
#             history_size=50,
#             tolerance_grad=1e-5,
#             tolerance_change=1.0 * np.finfo(float).eps,
#             line_search_fn="strong_wolfe"
#         )
#         self.iter = 0
#         self.loss = 0
#         for l in loss:
#             self.loss += l
#         self.model = model
#         self.epoches = epoches
#     def _loss_func(self):
#         self.optimizer2.zero_grad()
#         self.loss.backward()
#         self.iter += 1
#         if self.iter % 100 == 0:
#             print(f"epoch={self.iter},loss={self.loss}")
#         return self.loss
#     def train(self):
#         self.model.to(device)
#         self.model.train()
#         for epoch in range(epoches):
#             self.optimizer.zero_grad()
#             self.loss.backward()
#             self.optimizer.step()
#             if epoch % 100 == 0:
#                 print(f"epoch={epoch},loss={self.loss}")
#         self.optimizer2.step(self._loss_func)
#
# pinn = PINN(epoches, u_net, loss_interior, loss_all)
# pinn.train()

loss = loss_all + loss_interior

u_net.train()
for epoch in range(epoches):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch={epoch},loss={loss}")

# # Inference
# xc = torch.linspace(0, 1, h)
# xm, ym = torch.meshgrid(xc, xc)
# xx = xm.reshape(-1, 1)
# yy = ym.reshape(-1, 1)
# xy = torch.cat([xx, yy], dim=1)
# u_pred = u_net(xy)
# u_real = xx**2 * torch.exp(-yy)
# u_error = torch.abs(u_pred-u_real)
# u_pred_fig = u_pred.reshape(h,h)
# u_real_fig = u_real.reshape(h,h)
# u_error_fig = u_error.reshape(h, h)
# print( "Max abs error is: ", float(torch. max(torch. abs(u_pred - xx * xx * torch.exp(-yy)))))
# #仅有PDE损失
# print(xx)
# print(xy)
# #作PINN数值解图
# fig = plt. figure(1)
# ax = Axes3D(fig)
# ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_pred_fig. detach(). numpy())
# ax.text2D(0.5, 0.9, "PINN", transform=ax.transAxes)
# plt.show()
# fig.savefig( "PINN solve.png")