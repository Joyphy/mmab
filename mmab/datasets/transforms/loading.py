import numpy as np

import mmcv
import mmengine.fileio as fileio
from mmab.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class LoadMaskFromFile(BaseTransform):
    def transform(self, results):
        if "gt_seg_map" in results:
            if results["gt_seg_map"] is not None:
                results['gt_seg_map'] = mmcv.imfrombytes(fileio.get(results['gt_seg_map']), flag='unchanged', backend="cv2").squeeze()/np.array(255, dtype=np.float32)
            else:
                results['gt_seg_map'] = np.zeros(results["img_shape"], dtype=np.float32)
        return results