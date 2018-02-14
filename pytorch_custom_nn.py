"""
subclass nn 里面的一下函数，从而使得自己的模型更多样灵活
"""
import torch
from torch.autograd import Variable


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中定义模型结构
        :param D_in: 输入
        :param H: 隐藏单元
        :param D_out: 输出
        :return:
        """
        super(TwoLayerNet, self).__init__()
        self.liner1 = torch.nn.Linear(D_in, H)
        self.liner2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.liner1(x).clamp(min=0)
        y_pred = self.liner2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(t, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
