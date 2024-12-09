import torch

# 创建一个tensor，并设置requires_grad=True以追踪梯度
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

b = torch.tensor([4.0, 5.0, 6.0])

c = (a + b) / 10
# 计算y作为a的第一个元素
print(torch.mul(c[0], b))
y = torch.tensor([
  c[0] * 12,
  c[1] * c[1], 
  c[2] * c[0]
])

# 执行一些操作，例如计算y的平方
z = y**2 

# 调用backward()来计算梯度
z.sum().backward()

# 打印a的梯度
print(a.grad)  # 输出: tensor([2., 0., 0.])