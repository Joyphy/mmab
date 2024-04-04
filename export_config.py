"""
将base机制的链接config转换成一个完整config
"""

import os
from mmengine.config import Config

cfg_path = "/root/corespace/mmdeploy/configs/mmdet/detection/detection_tensorrt_static-800x1344.py"
suffix = "_export"

new_cfg_path = os.path.join(os.path.dirname(cfg_path), os.path.basename(cfg_path)[:-3] + suffix + ".py")
mmcv_cfg = Config.fromfile(cfg_path)
mmcv_cfg.dump(new_cfg_path)
print(f"导出版配置文件保存到:\n{new_cfg_path}")