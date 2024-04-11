import torch
import numpy as np
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmdet.apis import init_detector

class PatchCoreTorch():
    def __init__(self, model_config, checkpoint, device="cuda:0") -> None:
        self.model = init_detector(model_config, checkpoint, device=device, palette=[])
        self.test_pipeline = self._get_test_pipeline(self.model)

    def __call__(self, image):
        result = self._inference(image)
        return result.image_score.item(), result.score_map.squeeze()

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