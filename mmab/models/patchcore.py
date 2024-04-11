import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from mmab.registry import MODELS
from mmengine.model import BaseModel
from mmengine.logging import print_log
from mmengine.structures import InstanceData

from ..utils import KCenterGreedy, cdist

@MODELS.register_module()
class PatchCore(BaseModel):
    def __init__(self, backbone: dict, data_preprocessor=None, init_cfg=None, test_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(backbone)
        self.register_buffer("memory_bank", torch.zeros(0))

        gaussian_kernel_, self.gaussian_radius = gaussian_kernel(sigma=4.0)
        self.register_buffer("gaussian_kernel", gaussian_kernel_)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        try:
            self.register_buffer("memory_bank", torch.zeros_like(state_dict["memory_bank"], device=self.memory_bank.device))
        except:
            self.register_buffer("memory_bank", None)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def train_step(self, data):
        data = self.data_preprocessor(data, False)
        out = self._run_forward(data, mode="tensor")
        return out

    def val_step(self, data):
        data = self.data_preprocessor(data, False)
        dets, labels, masks = self._run_forward(data, mode="predict")
        score_map = masks.squeeze(1)
        image_score = dets[:, 0, 4]
        # score_map = postprocess_score_map(score_map, gaussian_blur=False) # 使用torch操作替代
        return InstanceData(
            score_map=score_map.cpu().numpy(),
            image_score=image_score.cpu().numpy()
        )

    def test_step(self, data):
        return self.val_step(data)

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        pool = torch.nn.AvgPool2d(3, 1, 1, count_include_pad=True)
        outputs = self.backbone(inputs)

        layer_indexs = [2, 3]
        x = []
        for i, l in enumerate(layer_indexs):
            if i != 0:
                x.append(F.interpolate(pool(outputs[l-1]), scale_factor=2**(layer_indexs[i] - layer_indexs[0]), mode="nearest"))
            else:
                x.append(pool(outputs[l-1]))
        x = torch.concat(x, 1)
        x = self.project(x)
        if mode == "predict":
            B, _, H, W = inputs.shape
            score_map, image_score = self.generate_scores_map(x, [H, W])
            dets = torch.full((B, 1, 5), 0.0, dtype=torch.float32, device=x.device)
            dets[..., 2] = W
            dets[..., 3] = H
            dets[..., 4] = image_score.unsqueeze(1)
            labels = torch.zeros((B, 1), dtype=torch.int32, device=x.device)
            score_map = self.postprocess_score_map_torch(score_map, gaussian_blur=True)
            return dets, labels, score_map
        return x
    
    def postprocess_score_map_torch(self, score_map, gaussian_blur=False):
        if gaussian_blur:
            padding = self.gaussian_radius
            score_map_padded = F.pad(score_map, torch.tensor([padding, padding, padding, padding], dtype=torch.int32), mode='reflect')
            blurred_score_map = F.conv2d(score_map_padded, self.gaussian_kernel)
            return blurred_score_map
        else:
            return score_map

    def project(self, x):
        return x

    def generate_scores_map(self, embedding, out_shape):
        # Nearest Neighbours distances
        B, C, H, W = embedding.shape
        embedding = embedding.permute((0, 2, 3, 1)).reshape((B, H * W, C))
        distances = self.nearest_neighbors(embedding=embedding, n_neighbors=9)
        distances = distances.permute((2, 0, 1))  # n_neighbors, B, HW
        image_score = []
        for i in range(B):
            image_score.append(
                self.compute_image_anomaly_score(distances[:, i, :]))
        distances = distances[0, :, :].reshape((B, H, W))

        score_map = F.interpolate(distances.unsqueeze(1), size=out_shape, mode='bilinear', align_corners=False)
        return score_map, torch.stack(image_score, dim=0)
    
    def nearest_neighbors(self, embedding, n_neighbors: int=9):
        """Compare embedding Features with the memory bank to get Nearest Neighbours distance
        """
        B, HW, C = embedding.shape
        n_coreset = self.memory_bank.shape[0]
        distances = []
        for i in range(B):
            distances.append(cdist(embedding[i, :, :], self.memory_bank, p=2.0))
        distances = torch.stack(distances, 0)
        # 使用PyTorch的topk获取最近的n_neighbors个邻居的距离
        distances, _ = torch.topk(distances, k=n_neighbors, dim=-1, largest=False, sorted=True)
        return distances  # 返回形状为[B, HW, n_neighbors]的张量

    @staticmethod
    def compute_image_anomaly_score(distance):
        """Compute Image-Level Anomaly Score for one nearest_neighbor distance map.
        """
        max_scores = torch.argmax(distance[0:1, :], dim=1)
        confidence = distance[:, max_scores]
        weights = 1 - (torch.max(torch.exp(confidence)) /
                    torch.sum(torch.exp(confidence)))
        score = weights * torch.max(distance[0, :])
        return score

    def compute_stats(self, embedding: torch.Tensor):
        C = embedding.shape[1]
        embedding = embedding.permute((0, 2, 3, 1)).reshape((-1, C))
        print_log(
            msg="Creating CoreSet Sampler via k-Center Greedy",
            logger="current")
        sampler = KCenterGreedy(embedding, sampling_ratio=10 / 100)
        print_log(
            msg="Getting the coreset from the main embedding.",
            logger="current")
        coreset = sampler.sample_coreset()
        print_log(
            msg=f"Assigning the coreset as the memory bank with shape {coreset.shape}.",
            logger="current")  # 18032,384
        self.register_buffer("memory_bank", coreset)

def postprocess_score_map(score_map, gaussian_blur=True):
    from scipy.ndimage import gaussian_filter
    score_map = score_map.cpu().numpy()
    
    if gaussian_blur:
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
    
    return score_map

def gaussian_kernel(sigma=1.0, truncate=4):
    radius = int(truncate * sigma + 0.5)
    size = 2 * radius + 1
    """生成高斯核"""
    axis = np.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = np.meshgrid(axis, axis)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return torch.tensor(kernel, dtype=torch.float32)[None, None, :, :], radius