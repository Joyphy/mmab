import torch

def cdist(X, Y, p=2.0):
    # 这个函数是核心的求特征向量距离的
    # 使用向量法更快, 但是容易爆显存
    # diff = X[:, None, :] - Y[None, :, :]
    # D = torch.norm(diff, p=p, dim=-1)
    if p == 2.0:
        return cdist_efficient(X, Y, p=2.0)
    else:
        D = []
        for i in range(len(X)):
            D.append(torch.norm(X[i, None, :] - Y, p=p, dim=-1))
        D = torch.stack(D, 0)
    return D

def cdist_efficient(X, Y, p=2.0):
    if p != 2.0:
        raise ValueError("This method only supports p=2.0 for Euclidean distance.")
    X_sq_sum = torch.sum(X**2, dim=1, keepdim=True)
    Y_sq_sum = torch.sum(Y**2, dim=1, keepdim=True)
    XY = torch.mm(X, Y.t())
    D_sq = X_sq_sum - 2*XY + Y_sq_sum.t()
    D = torch.sqrt(torch.clamp(D_sq, min=0.0))
    return D