import torch
import time

# 原始矩阵方法
def cdist_matrix(X, Y, p=2.0):
    diff = X[:, None, :] - Y[None, :, :]
    D = torch.norm(diff, p=p, dim=-1)
    return D

# 优化后的方法
def cdist_efficient(X, Y, p=2.0):
    if p != 2.0:
        raise ValueError("This method only supports p=2.0 for Euclidean distance.")
    X_sq_sum = torch.sum(X**2, dim=1, keepdim=True)
    Y_sq_sum = torch.sum(Y**2, dim=1, keepdim=True)
    XY = torch.mm(X, Y.t())
    D_sq = X_sq_sum - 2*XY + Y_sq_sum.t()
    D = torch.sqrt(torch.clamp(D_sq, min=0.0))
    return D

# 生成随机数据
X = torch.randn(1024, 384, dtype=torch.float32).to("cuda:0")
Y = torch.randn(4096, 384, dtype=torch.float32).to("cuda:0")
D_matrix = torch.cdist(X, Y, p=2.0)

# 测试原始方法
start_time = time.time()
D_matrix = torch.cdist(X, Y, p=2.0)
matrix_time = time.time() - start_time
time.sleep(10)

# 测试优化方法
start_time = time.time()
D_efficient = cdist_efficient(X, Y, p=2.0)
efficient_time = time.time() - start_time
time.sleep(10)

# 计算误差
error = torch.norm(D_matrix - D_efficient, p=2.0)

print(matrix_time, efficient_time, error)
print(D_matrix, D_efficient)
