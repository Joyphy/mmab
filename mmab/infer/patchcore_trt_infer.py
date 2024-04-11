import numpy as np
from mmdeploy_runtime import Detector
from scipy.ndimage import gaussian_filter

class PatchCoreTRT():
    def __init__(self, model_path, device_name="cuda", device_id=0, norm=True, gaussian_blur=True, sigma=4) -> None:
        self.norm = norm
        self.sigma = sigma
        self.gaussian_blur = gaussian_blur
        self.model = Detector(model_path=model_path, device_name=device_name, device_id=device_id)

    def __call__(self, image):
        dets, _, masks = self.model(image)
        h, w = masks[0].shape
        score_map = masks[0].reshape(h, w//4, 4).view(np.float32).squeeze()
        if self.gaussian_blur:
            score_map = self._blur(score_map)
        if self.norm:
            score_map = self._norm(score_map)
        image_score = dets[0, 4].item()
        return image_score, score_map

    def _blur(self, score_map):
        return gaussian_filter(score_map, sigma=self.sigma)

    def _norm(self, score_map):
        max_v, min_v = score_map.max(), score_map.min()
        score_map = (score_map - min_v) / (max_v - min_v)
        return score_map