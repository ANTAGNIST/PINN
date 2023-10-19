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
# 创建tensor_cuda数组，require_grad==True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_cuda = dict(data)
for key, value in data_cuda.items():
    data_cuda[key] = value.clone().to(device)
# 创建ndarray数组
data_ndarray = dict(data)
for key, value in data_ndarray.items():
    data_ndarray[key] = value.detach().cpu().numpy()

# 创建损失函数
loss_fn = torch.nn.MSELoss()
# PDE损失
loss_interior = data_cuda["d2u(x,y)_dx2N"] - data_cuda["d4u(x,y)_dx4N"] - (2-data_cuda["xN"]) * torch.exp(-data_cuda["yN"])
loss_down_yy = data_cuda["d2u(x,0)_dy2N1"] - data_cuda["xN1"]**2
loss_up_yy = data_cuda["d2u(x,1)_dy2N1"] - data_cuda["xN1"]**2/torch.e
loss_down = data_cuda["u(x,0)N1"] - data_cuda["xN1"]
loss_up = data_cuda["u(x,1)N1"] - data_cuda["xN1"]**2/torch.e
loss_left = data_cuda["u(0,y)N1"]
loss_right = data_cuda["u(1,y)N1"] - torch.exp(-data_cuda["yN1"])
loss_all = loss_interior + loss_down_yy + loss_up_yy + loss_down + loss_up + loss_left + loss_right
# 创建优化器
optimizer = torch.optim.Adam(u_net.parameters(), lr=0.001)
optimizer2 = torch.optim.LBFGS(
        u_net.parameters(),
        lr=1.0,
        max_iter=50000,
        max_eval=50000,
        history_size=50,
        tolerance_grad=1e-5,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe"
)
# optimizer2需要传递一个闭包参数才能运行
# https://blog.csdn.net/awslyyds/article/details/127503167
def loss_func():
    optimizer2.zero_grad()
    loss = loss_fn(loss_all, )
    u_pred = net_u(x_u, t_u)
    f_pred = net_f(x_f, t_f)
    loss_u = torch.mean((self.u - u_pred) ** 2)
    loss_f = torch.mean(f_pred ** 2)

    loss = loss_u + loss_f

    loss.backward()
    self.iter += 1
    if self.iter % 100 == 0:
        print(
            'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
        )
    return loss
def _train(model, loss, epoches=epoches):
    model.to(device)
    for epoch in range(epoches):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch={epoch}, loss=[{loss}]")

u_net.train()
_train()

# Inference
xc = torch.linspace(0, 1, h)
xm, ym = torch.meshgrid(xc, xc)
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
u_pred = model(xy)
u_real = xx**2 * torch.exp(-yy)
u_error = torch.abs(u_pred-u_real)
u_pred_fig = u_pred.reshape(h,h)
u_real_fig = u_real.reshape(h,h)
u_error_fig = u_error.reshape(h, h)
print( "Max abs error is: ", float(torch. max(torch. abs(u_pred - xx * xx * torch.exp(-yy)))))
#仅有PDE损失
print(xx)
print(xy)
#作PINN数值解图
fig = plt. figure(1)
ax = Axes3D(fig)
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_pred_fig. detach(). numpy())
ax.text2D(0.5, 0.9, "PINN", transform=ax.transAxes)
plt.show()
fig.savefig( "PINN solve.png")

def _loss_func(optimizer, *loss):
    iter = 0
    optimizer.zero_grad()
    loss = sum(loss)
    loss.backward()
    iter += 1
    if iter % 100 == 0:
        print(f"epoch={iter},loss={loss}")
    return loss
def train(epoches, model, optimizer, optimizer2, loss_fn, *loss):
    model.to(device)
    model.train()
    for epoch in range(epoches):
        loss = sum(loss)
        optimizer.step(loss_fn)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch={iter},loss={loss}")
    optimizer2.step(loss_fn)

# 创建优化器
optimizer = torch.optim.Adam(u_net.parameters(), lr=0.001)
optimizer2 = torch.optim.LBFGS(
        u_net.parameters(),
        lr=1.0,
        max_iter=50000,
        max_eval=50000,
        history_size=50,
        tolerance_grad=1e-5,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe"
)