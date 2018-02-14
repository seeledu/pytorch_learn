"""
在nnPackage的基础上，应用optimPackage来进行梯度下降
"""
import torch
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)
loss_fn =torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    print(t,loss.data[0])
    # 在进行梯度下降之前，要先将缓存的梯度的数据清空
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()