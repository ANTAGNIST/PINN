import torch
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
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 内点
def interior(n=N):
    x = torch.randn(n, 1)
    y = torch.randn(n, 1)
    cond = (2 - x**2)*torch.exp(-y)
    return x.requires_grad_(True).to(device), y.requires_grad_(True).to(device), cond.to(device)
# 边界 d2u_dy2(x,0)=x^2
def down_yy(n=N1):
    x = torch.randn(n, 1)
    y = torch.zeros_like(x)
    cond = x**2
    return x.requires_grad_(True).to(device), y.requires_grad_(True).to(device), cond.to(device)
# 边界 d2u_dy2(x,1)=x^2/e
def up_yy(n=N1):
    x = torch.randn(n, 1)
    y = torch.ones_like(x)
    cond = x**2/torch.e
    return x.requires_grad_(True).to(device), y.requires_grad_(True).to(device), cond.to(device)
# 边界 u(x,0)=x^2
def down(n=N1):
    x = torch.randn(n, 1)
    y = torch.zeros_like(x)
    cond = x**2
    return x.requires_grad_(True).to(device), y.requires_grad_(True).to(device), cond.to(device)
# 边界 u(x,1)=x^2/e
def up(n=N1):
    x = torch.randn(n, 1)
    y = torch.ones_like(x)
    cond = x**2/torch.e
    return x.requires_grad_(True).to(device), y.requires_grad_(True).to(device), cond.to(device)
# 边界 u(0, y)=0
def left(n=N1):
    y = torch.randn(n, 1)
    x = torch.zeros_like(y)
    cond = torch.zeros_like(y)
    return x.requires_grad_(True).to(device), y.requires_grad_(True).to(device), cond.to(device)
# 边界 u(1, y)=e^(-y)
def right(n=N1):
    y = torch.randn(n, 1)
    x = torch.ones_like(y)
    cond = torch.zeros_like(y)
    return x.requires_grad_(True).to(device), y.requires_grad_(True).to(device), cond.to(device)
# r.s.内点
def data_interior(n=N2):
    y = torch.randn(n, 1)
    x = torch.randn(n, 1)
    cond = x**2 * torch.exp(-y)
    return x.requires_grad_(True).to(device), y.requires_grad_(True).to(device), cond.to(device)

loss = torch.nn.MSELoss()
# PDE损失
def l_interior(u):
    x, y, r_eq = interior()
    uxy = u(torch.cat([x, y], dim=1))
    l_eq = gradient(uxy, x, 2) - gradient(uxy, y, 4)
    return loss(l_eq, r_eq)
def l_down_yy(u):
    x, y, r_eq = down_yy()
    uxy = u(torch.cat([x, y], dim=1))
    l_eq = gradient(uxy, x, 2)
    return loss(l_eq, r_eq)
def l_up_yy(u):
    x, y, r_eq = up_yy()
    uxy = u(torch.cat([x, y], dim=1))
    l_eq = gradient(uxy, x, 2)
    return loss(l_eq, r_eq)
def l_down(u):
    x, y, r_eq = down()
    uxy = u(torch.cat([x, y], dim=1))
    l_eq = uxy
    return loss(l_eq, r_eq)
def l_up(u):
    x, y, r_eq = up()
    uxy = u(torch.cat([x, y], dim=1))
    l_eq = uxy
    return loss(l_eq, r_eq)
def l_left(u):
    x, y, r_eq = left()
    uxy = u(torch.cat([x, y], dim=1))
    l_eq = uxy
    return loss(l_eq, r_eq)
def l_right(u):
    x, y, r_eq = right()
    uxy = u(torch.cat([x, y], dim=1))
    l_eq = uxy
    return loss(l_eq, r_eq)
# 构造数据损失
def l_data(u):
    x, y, r_eq = data_interior()
    uxy = u(torch.cat([x, y], dim=1))
    l_eq = uxy
    return loss(l_eq, r_eq)
def l_all(u, *args):
    temp = 0
    for l in args:
        temp = temp + l(u)
    return temp
model = MLP(2, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def _train(epoches=epoches):
    model.to(device)
    for epoch in range(epoches):
        optimizer.zero_grad()
        l = l_all(model, l_interior, l_down_yy, l_up_yy, l_down, l_up, l_left, l_right)
        l.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch={epoch}, loss=[{l}]")

model.train()
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