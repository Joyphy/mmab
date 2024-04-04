import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """生成高斯核"""
    axis = np.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = np.meshgrid(axis, axis)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return torch.tensor(kernel, dtype=torch.float32)

def gaussian_blur_torch(image, sigma=1.0, truncate=4.0):
    """使用PyTorch进行高斯模糊，包括反射填充"""
    radius = int(truncate * sigma + 0.5)
    size = 2 * radius + 1
    kernel = gaussian_kernel(size, sigma).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(image.shape[0], 1, 1, 1)

    # 应用反射填充
    padding = radius
    image_padded = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
    blurred_image = F.conv2d(image_padded, kernel, groups=image.shape[0])
    
    return blurred_image.squeeze(0)

# 创建测试图像
np.random.seed(0)  # 为了可重复性
test_image = a = np.arange(98, step=2).reshape((7,7)).astype(np.float32)
test_image_torch = torch.from_numpy(test_image).unsqueeze(0)  # PyTorch期望通道在前

# 应用SciPy的高斯模糊
blurred_image_scipy = gaussian_filter(test_image, sigma=1)
print(test_image_torch.shape)

# 应用自定义的PyTorch高斯模糊
blurred_image_torch = gaussian_blur_torch(test_image_torch, sigma=1).numpy()
print(blurred_image_scipy)
print(blurred_image_torch)

# 计算误差
error = np.abs(blurred_image_scipy - blurred_image_torch).mean()
print(f'Mean absolute error between SciPy and PyTorch: {error}')
