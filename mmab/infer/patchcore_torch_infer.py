import torch
from mmcv import imresize
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmdet.apis import init_detector

class PatchCoreTorch():
    def __init__(self, model_config, checkpoint, device="cuda:0", norm=False) -> None:
        self.norm = norm
        self.model = init_detector(model_config, checkpoint, device=device, palette=[])
        self.is_resize_mask_ = self.model.cfg["model"]["test_cfg"]["is_resize_mask"]
        self.test_pipeline = self._get_test_pipeline(self.model)

    def __call__(self, image):
        result = self._inference(image)
        image_score = result.image_score.item()
        score_map = result.score_map.squeeze()
        if self.is_resize_mask_ and score_map.shape != image.shape[:2]:
            score_map = imresize(score_map, image.shape[:2], backend="cv2")
        if self.norm:
            score_map = self._norm(score_map)
        return image_score, score_map

    def _inference(self, image):
        data_ = dict(img=image, img_id=0)
        data_ = self.test_pipeline(data_)
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        with torch.no_grad():
            results = self.model.test_step(data_)[0]
        return results
    
    def _get_test_pipeline(self, model):
        cfg = model.cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(test_pipeline)
        return test_pipeline

    def _norm(self, score_map):
        max_v, min_v = score_map.max(), score_map.min()
        score_map = (score_map - min_v) / (max_v - min_v)
        return score_map